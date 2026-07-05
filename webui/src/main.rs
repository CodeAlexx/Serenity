// serenity-web-trainer — the web supervisor for the Mojo trainer fleet.
// Build spec: docs/UI_MAP_2026-07-05.md §9. Replaces the X11 Mojo UI's launch
// path: no gnome-terminal, no DISPLAY coupling; direct child spawn with the
// run's log in its own workspace. Presets are DATA (presets.json), every row
// complete (fixes the Mojo UI's cross-preset naming contamination, §8.1).

use axum::{
    extract::{Path as AxPath, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{Child, Command},
    sync::{broadcast, Mutex},
};

const REPO_ROOT: &str = "/home/alex/serenity-trainer";
const CUDA_LD: &str = "/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib";

#[derive(Clone, Serialize, Deserialize)]
struct Preset {
    id: String,
    label: String,
    backend: String,
    wired: bool,
    argv_shape: String,
    binary: String,
    base_config: String,
    checkpoint: String,
    cache: String,
    run_name: String,
    workspace_dir: String,
    save_filename_prefix: String,
    recipe: Value,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[derive(Serialize, Clone)]
struct RunInfo {
    id: u64,
    preset_id: String,
    backend: String,
    workspace_dir: String,
    log_path: String,
    status: String, // running | exited | stopped | failed
    pid: Option<u32>,
    // live parsed progress (UI_MAP §5: both line shapes)
    step: u64,
    total_steps: u64,
    epoch: u64,
    total_epochs: u64,
    loss: f64,
    grad_norm: f64,
    s_per_step: f64,
    eta: String,
}

struct AppState {
    presets: Vec<Preset>,
    runs: Mutex<Vec<RunInfo>>,
    child: Mutex<Option<Child>>, // one run at a time (the 24GB rule)
    events: broadcast::Sender<String>,
    next_id: Mutex<u64>,
}

#[derive(Deserialize)]
struct LaunchReq {
    preset_id: String,
    /// recipe overrides merged over the preset recipe (lr, rank, steps, cadences, ...)
    #[serde(default)]
    overrides: Value,
    #[serde(default)]
    cache: Option<String>,
    #[serde(default)]
    run_name: Option<String>,
    /// FULL resume: path to the .state file (MJ-1077 — the .state PATH, not the PEFT)
    #[serde(default)]
    resume_state: Option<String>,
    #[serde(default)]
    start_step: Option<u64>,
    /// build the config + argv and return them WITHOUT spawning (wiring verification)
    #[serde(default)]
    dry_run: bool,
}

fn gpu_busy() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // ignore idle stubs under 1GB (e.g. a parked ComfyUI holding ~300MB)
    let heavy: Vec<&str> = s
        .lines()
        .filter(|l| {
            l.rsplit(',')
                .next()
                .and_then(|m| m.trim().split(' ').next())
                .and_then(|n| n.parse::<u64>().ok())
                .map(|mib| mib > 1024)
                .unwrap_or(true)
        })
        .collect();
    if heavy.is_empty() { None } else { Some(heavy.join("; ")) }
}

/// Parse the two accepted progress line shapes (UI_MAP §5) with plain string ops.
fn parse_progress(line: &str, run: &mut RunInfo) -> bool {
    // shape 2 (klein/krea2/tqdm): "[X] step 1613/2000 | epoch 14/17 | loss 0.59 | grad_norm 0.15 | 2.1s/step | ... | ETA 0:13:20"
    let grab = |key: &str| -> Option<String> {
        let i = line.find(key)? + key.len();
        let rest = &line[i..];
        let end = rest.find(|c: char| c == '|').unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    };
    let mut hit = false;
    if let Some(sv) = grab("step ") {
        if let Some((a, b)) = sv.split_once('/') {
            if let (Ok(a), Ok(b)) = (a.trim().parse(), b.trim().split(' ').next().unwrap_or("0").parse()) {
                run.step = a;
                run.total_steps = b;
                hit = true;
            }
        }
    }
    if let Some(ev) = grab("epoch ") {
        if let Some((a, b)) = ev.split_once('/') {
            if let (Ok(a), Ok(b)) = (a.trim().parse(), b.trim().split(' ').next().unwrap_or("0").parse()) {
                run.epoch = a;
                run.total_epochs = b;
            }
        }
    }
    if let Some(l) = grab("loss ") {
        if let Ok(v) = l.split(' ').next().unwrap_or("").parse() {
            run.loss = v;
        }
    }
    if let Some(g) = grab("grad_norm ") {
        if let Ok(v) = g.split(' ').next().unwrap_or("").parse() {
            run.grad_norm = v;
        }
    }
    if let Some(i) = line.find("s/step") {
        let head = &line[..i];
        if let Some(tok) = head.rsplit(|c: char| c == ' ' || c == '|').next() {
            if let Ok(v) = tok.trim().parse() {
                run.s_per_step = v;
            }
        }
    }
    if let Some(i) = line.find("ETA ") {
        run.eta = line[i + 4..].trim().to_string();
    }
    hit
}

async fn launch(State(st): State<Arc<AppState>>, Json(req): Json<LaunchReq>) -> (StatusCode, Json<Value>) {
    let Some(p) = st.presets.iter().find(|p| p.id == req.preset_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": format!("unknown preset {}", req.preset_id)})));
    };
    if !p.wired {
        return (StatusCode::NOT_IMPLEMENTED, Json(json!({"error": format!("backend {} not wired in the web supervisor yet", p.backend)})));
    }
    if st.child.lock().await.is_some() {
        return (StatusCode::CONFLICT, Json(json!({"error": "a run is already active (one at a time)"})));
    }
    if let Some(who) = gpu_busy() {
        return (StatusCode::CONFLICT, Json(json!({"error": format!("GPU busy: {who}")})));
    }
    // ---- build the runner config: base template + preset recipe + overrides ----
    let mut cfg: Value = if p.base_config.is_empty() {
        json!({}) // ideogram4: all-argv contract, no runner config template
    } else {
        let base_path = Path::new(REPO_ROOT).join(&p.base_config);
        match std::fs::read_to_string(&base_path).ok().and_then(|s| serde_json::from_str(&s).ok()) {
            Some(v) => v,
            None => return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": format!("base config missing/unreadable: {} — run this model once via CLI or add the template", base_path.display())}))),
        }
    };
    let run_name = req.run_name.clone().unwrap_or_else(|| p.run_name.clone());
    let workspace = format!("/home/alex/mojodiffusion/output/{run_name}");
    if let Value::Object(m) = &mut cfg {
        if let Value::Object(r) = &p.recipe {
            for (k, v) in r {
                m.insert(k.clone(), v.clone());
            }
        }
        if let Value::Object(o) = &req.overrides {
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
        m.insert("workspace_dir".into(), json!(workspace));
        m.insert("save_filename_prefix".into(), json!(run_name));
    }
    let cache = req.cache.clone().unwrap_or_else(|| p.cache.clone());
    if p.argv_shape == "krea2" && !Path::new(&cache).exists() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": format!("cache not found: {cache}")})));
    }
    std::fs::create_dir_all(format!("{workspace}/samples")).ok();
    let cfg_path = format!("{workspace}/web_train_config.json");
    if std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap()).is_err() {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "cannot write runner config"})));
    }
    let steps = cfg.get("max_steps").and_then(|v| v.as_u64()).unwrap_or(2000);
    let getf = |k: &str, d: f64| cfg.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    let getu = |k: &str, d: u64| cfg.get(k).and_then(|v| v.as_u64()).unwrap_or(d);
    // ---- all 4 argv shapes (UI_MAP §4) + the resume slot the Mojo UI never wired ----
    let mut args: Vec<String> = match p.argv_shape.as_str() {
        "krea2" => vec![cache.clone(), steps.to_string(), cfg_path.clone()],
        "config_runner" => vec![cfg_path.clone(), steps.to_string()],
        // shape 3: <stage_dir> <steps> <lr> <rank> <out_dir> - <config.json>
        // (argv wins for steps/lr/rank/out_dir; "-" keeps EMA config-owned)
        "hidream" => vec![
            cache.clone(),
            steps.to_string(),
            format!("{}", getf("learning_rate", 1e-4)),
            format!("{}", getu("lora_rank", 16)),
            workspace.clone(),
            "-".into(),
            cfg_path.clone(),
        ],
        // shape 4: the 18-arg ideogram4 line (TrainerRuntimeBridge.mojo:354-403).
        // argv 11 levers "-" (defaults stay byte-identical, C13), argv 16 resume "-",
        // argv 17 prompt "-" (no inline sampling in unit 2), argv 18 resolution.
        "ideogram4" => vec![
            format!("{workspace}/progress.log"),
            format!("{}/transformer/diffusion_pytorch_model.safetensors", p.checkpoint),
            cache.clone(),
            "/home/alex/mojodiffusion/output".into(),
            steps.to_string(),
            format!("{}", getu("lora_rank", 16)),
            format!("{}", getf("lora_alpha", 16.0)),
            format!("{}", getf("learning_rate", 1e-4)),
            format!("{}", getu("save_every", 500)),
            format!("{}", getf("caption_dropout", 0.0)),
            "-".into(),
            format!("{}", getu("sample_every", 0)),
            "20".into(),
            "4.5".into(),
            format!("{}", getu("seed", 42)),
            "-".into(),
            "-".into(),
            "512".into(),
        ],
        other => return (StatusCode::NOT_IMPLEMENTED, Json(json!({"error": format!("argv shape {other} not wired")}))),
    };
    if let Some(state) = &req.resume_state {
        if p.argv_shape != "krea2" && p.argv_shape != "config_runner" {
            return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "resume is wired for krea2/config-runner shapes only"})));
        }
        if !state.ends_with(".state") && !state.contains(".state.") {
            return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "resume_state must be the .state file path (MJ-1077: the PEFT path silently warm-resumes)"})));
        }
        args.push(state.clone());
        args.push(req.start_step.unwrap_or(0).to_string());
    }
    if req.dry_run {
        return (StatusCode::OK, Json(json!({"dry_run": true, "binary": p.binary, "args": args, "config_written": cfg_path, "config": cfg})));
    }
    let log_path = format!("{workspace}/train_web.log");
    let log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("log create: {e}")}))),
    };
    // stdbuf -oL: the child now writes to a PIPE (no tty), where libc stdio
    // block-buffers ~4-8KB — without line buffering, SSE events arrive in
    // delayed bursts instead of live lines.
    let mut cmd = Command::new("stdbuf");
    cmd.arg("-oL")
        .arg("-eL")
        .arg(Path::new(REPO_ROOT).join(&p.binary))
        .args(&args)
        .current_dir(REPO_ROOT)
        .env("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "true")
        .env("LD_LIBRARY_PATH", CUDA_LD)
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log_file.try_clone().unwrap()));
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("spawn: {e}")}))),
    };
    let pid = child.id();
    let id = {
        let mut n = st.next_id.lock().await;
        *n += 1;
        *n
    };
    let info = RunInfo {
        id,
        preset_id: p.id.clone(),
        backend: p.backend.clone(),
        workspace_dir: workspace.clone(),
        log_path: log_path.clone(),
        status: "running".into(),
        pid,
        step: 0,
        total_steps: steps,
        epoch: 0,
        total_epochs: 0,
        loss: 0.0,
        grad_norm: 0.0,
        s_per_step: 0.0,
        eta: String::new(),
    };
    st.runs.lock().await.push(info.clone());
    // pump stdout -> logfile + parse + broadcast SSE events
    let stdout = child.stdout.take().unwrap();
    *st.child.lock().await = Some(child);
    let stc = st.clone();
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        let mut lf = std::fs::OpenOptions::new().append(true).open(&log_path).ok();
        while let Ok(Some(line)) = lines.next_line().await {
            use std::io::Write;
            if let Some(f) = lf.as_mut() {
                let _ = writeln!(f, "{line}");
            }
            let mut runs = stc.runs.lock().await;
            if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
                let parsed = parse_progress(&line, r);
                let ev = if parsed {
                    json!({"type": "progress", "run": &*r}).to_string()
                } else {
                    json!({"type": "log", "run_id": id, "line": line}).to_string()
                };
                let _ = stc.events.send(ev);
            }
        }
        // child stdout closed -> reap
        let status = {
            let mut ch = stc.child.lock().await;
            let s = match ch.as_mut() {
                Some(c) => c.wait().await.ok(),
                None => None,
            };
            *ch = None;
            s
        };
        let mut runs = stc.runs.lock().await;
        if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
            if r.status == "running" {
                r.status = match status {
                    Some(s) if s.success() => "exited".into(),
                    _ => "failed".into(),
                };
            }
            append_history(r);
            let _ = stc.events.send(json!({"type": "status", "run": &*r}).to_string());
        }
    });
    (StatusCode::OK, Json(json!({"run_id": id, "workspace": workspace, "log": info.log_path, "tail": format!("tail -f {}", info.log_path)})))
}

async fn stop(State(st): State<Arc<AppState>>, AxPath(id): AxPath<u64>) -> (StatusCode, Json<Value>) {
    let mut ch = st.child.lock().await;
    if let Some(c) = ch.as_mut() {
        if let Some(pid) = c.id() {
            unsafe { libc_kill(pid as i32, 15) };
            tokio::time::sleep(Duration::from_secs(2)).await;
            let _ = c.start_kill();
        }
    }
    *ch = None;
    let mut runs = st.runs.lock().await;
    if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
        r.status = "stopped".into();
        let _ = st.events.send(json!({"type": "status", "run": &*r}).to_string());
    }
    (StatusCode::OK, Json(json!({"stopped": id})))
}

extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

async fn list_runs(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"runs": &*st.runs.lock().await, "gpu_busy": gpu_busy()}))
}

async fn system_metrics() -> Json<Value> {
    // GPU via nvidia-smi (the Mojo UI used a MojoUI backend API — UI_MAP §5)
    let gpu = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version,utilization.gpu,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let g: Vec<String> = gpu.split(',').map(|s| s.trim().to_string()).collect();
    // RAM from /proc/meminfo
    let mem = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let memkb = |key: &str| -> u64 {
        mem.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let total = memkb("MemTotal:");
    let avail = memkb("MemAvailable:");
    Json(json!({
        "gpu_name": g.first().cloned().unwrap_or_default(),
        "gpu_driver": g.get(1).cloned().unwrap_or_default(),
        "gpu_util": g.get(2).cloned().unwrap_or_default(),
        "gpu_temp": g.get(3).cloned().unwrap_or_default(),
        "vram_used_mb": g.get(4).cloned().unwrap_or_default(),
        "vram_total_mb": g.get(5).cloned().unwrap_or_default(),
        "ram_used_gb": format!("{:.1}", (total.saturating_sub(avail)) as f64 / 1048576.0),
        "ram_total_gb": format!("{:.1}", total as f64 / 1048576.0),
    }))
}

async fn get_presets(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"presets": st.presets}))
}

async fn samples(State(st): State<Arc<AppState>>, AxPath(id): AxPath<u64>) -> Json<Value> {
    let runs = st.runs.lock().await;
    let Some(r) = runs.iter().find(|r| r.id == id) else {
        return Json(json!({"error": "unknown run"}));
    };
    let mut out = vec![];
    for sub in ["samples", "turbo_samples"] {
        let dir = format!("{}/{sub}", r.workspace_dir);
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if let Ok(rd2) = std::fs::read_dir(&p) {
                        for e2 in rd2.flatten() {
                            if e2.path().extension().map(|x| x == "png").unwrap_or(false) {
                                out.push(e2.path().to_string_lossy().to_string());
                            }
                        }
                    }
                } else if p.extension().map(|x| x == "png").unwrap_or(false) {
                    out.push(p.to_string_lossy().to_string());
                }
            }
        }
    }
    out.sort();
    Json(json!({"samples": out.iter().map(|p| format!("/files{p}")).collect::<Vec<_>>()}))
}

fn media_ok(p: &str) -> bool {
    // read-only serving scope: user-owned media/text under /home/alex, no traversal, no dotfiles
    p.starts_with("/home/alex/")
        && !p.contains("..")
        && !p.split('/').any(|seg| seg.starts_with('.') && seg.len() > 2)
        && [".png", ".jpg", ".jpeg", ".webp", ".txt", ".json"].iter().any(|e| p.to_lowercase().ends_with(e))
}

async fn file_serve(AxPath(path): AxPath<String>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let full = format!("/{path}");
    if !media_ok(&full) {
        return StatusCode::FORBIDDEN.into_response();
    }
    match std::fs::read(&full) {
        Ok(bytes) => {
            let low = full.to_lowercase();
            let ct = if low.ends_with(".png") { "image/png" }
                else if low.ends_with(".jpg") || low.ends_with(".jpeg") { "image/jpeg" }
                else if low.ends_with(".webp") { "image/webp" }
                else if low.ends_with(".json") { "application/json" }
                else { "text/plain; charset=utf-8" };
            ([(axum::http::header::CONTENT_TYPE, ct)], bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[derive(Deserialize)]
struct PathQ { path: String }

/// Dataset browser: images in a folder + caption sidecar presence (UI_MAP §6/§7)
async fn dataset_media(axum::extract::Query(q): axum::extract::Query<PathQ>) -> Json<Value> {
    if !q.path.starts_with("/home/alex/") || q.path.contains("..") {
        return Json(json!({"error": "path out of scope"}));
    }
    let mut items = vec![];
    if let Ok(rd) = std::fs::read_dir(&q.path) {
        let mut files: Vec<_> = rd.flatten().map(|e| e.path()).collect();
        files.sort();
        for p in files {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            if ["png", "jpg", "jpeg", "webp"].contains(&ext.as_str()) {
                let cap = p.with_extension("txt");
                items.push(json!({
                    "image": format!("/files{}", p.display()),
                    "path": p.to_string_lossy(),
                    "caption_path": cap.exists().then(|| cap.to_string_lossy().to_string()),
                }));
            }
        }
    }
    Json(json!({"count": items.len(), "items": items}))
}

/// Caption sidecar read/write — string-native (kills the Mojo CaptionerTab byte-slice crash class)
async fn caption_get(axum::extract::Query(q): axum::extract::Query<PathQ>) -> Json<Value> {
    if !q.path.starts_with("/home/alex/") || !q.path.ends_with(".txt") || q.path.contains("..") {
        return Json(json!({"error": "caption path out of scope"}));
    }
    Json(json!({"path": q.path, "text": std::fs::read_to_string(&q.path).unwrap_or_default()}))
}

#[derive(Deserialize)]
struct CaptionPut { path: String, text: String }

async fn caption_put(Json(b): Json<CaptionPut>) -> (StatusCode, Json<Value>) {
    if !b.path.starts_with("/home/alex/") || !b.path.ends_with(".txt") || b.path.contains("..") {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "caption path out of scope"})));
    }
    match std::fs::write(&b.path, &b.text) {
        Ok(_) => (StatusCode::OK, Json(json!({"saved": b.path}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    }
}

/// Validation prompts (samples JSON) editor
async fn validations_get(axum::extract::Query(q): axum::extract::Query<PathQ>) -> Json<Value> {
    if !q.path.ends_with(".json") || q.path.contains("..") {
        return Json(json!({"error": "json path required"}));
    }
    match std::fs::read_to_string(&q.path).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
        Some(v) => Json(json!({"path": q.path, "content": v})),
        None => Json(json!({"error": format!("cannot read/parse {}", q.path)})),
    }
}

#[derive(Deserialize)]
struct ValidationsPut { path: String, content: Value }

async fn validations_put(Json(b): Json<ValidationsPut>) -> (StatusCode, Json<Value>) {
    if !b.path.starts_with("/home/alex/") || !b.path.ends_with(".json") || b.path.contains("..") {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "path out of scope"})));
    }
    // enforce the user's 1024 minimum for image validation renders (samples validator rule)
    if let Some(d) = b.content.get("defaults") {
        let w = d.get("width").and_then(|v| v.as_u64()).unwrap_or(1024);
        let h = d.get("height").and_then(|v| v.as_u64()).unwrap_or(1024);
        let enforce = d.get("enforce_min_image_size").and_then(|v| v.as_bool()).unwrap_or(true);
        if enforce && (w < 1024 || h < 1024) {
            return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "image validation samples must be 1024x1024 or larger (user standard)"})));
        }
    }
    match std::fs::write(&b.path, serde_json::to_string_pretty(&b.content).unwrap()) {
        Ok(_) => (StatusCode::OK, Json(json!({"saved": b.path}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    }
}

fn history_path() -> String {
    format!("{REPO_ROOT}/webui/runs_history.jsonl")
}

fn append_history(r: &RunInfo) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(history_path()) {
        let _ = writeln!(f, "{}", serde_json::to_string(r).unwrap_or_default());
    }
}

async fn runs_history() -> Json<Value> {
    let s = std::fs::read_to_string(history_path()).unwrap_or_default();
    let rows: Vec<Value> = s.lines().rev().take(50).filter_map(|l| serde_json::from_str(l).ok()).collect();
    Json(json!({"history": rows}))
}

async fn events(State(st): State<Arc<AppState>>) -> Sse<EvStream> {
    let rx = st.events.subscribe();
    Sse::new(EvStream::new(rx)).keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(15)))
}

// broadcast -> Stream adapter: the receiver round-trips THROUGH the future's
// output so ownership stays sound without extra stream crates.
type RecvFut = std::pin::Pin<Box<dyn std::future::Future<Output = (Option<String>, broadcast::Receiver<String>)> + Send>>;

async fn recv_owned(mut rx: broadcast::Receiver<String>) -> (Option<String>, broadcast::Receiver<String>) {
    loop {
        match rx.recv().await {
            Ok(m) => return (Some(m), rx),
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => return (None, rx),
        }
    }
}

struct EvStream {
    fut: RecvFut,
}
impl EvStream {
    fn new(rx: broadcast::Receiver<String>) -> Self {
        Self { fut: Box::pin(recv_owned(rx)) }
    }
}
impl futures_core::Stream for EvStream {
    type Item = Result<Event, std::convert::Infallible>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.fut.as_mut().poll(cx) {
            std::task::Poll::Ready((Some(m), rx)) => {
                self.fut = Box::pin(recv_owned(rx));
                std::task::Poll::Ready(Some(Ok(Event::default().data(m))))
            }
            std::task::Poll::Ready((None, _)) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[tokio::main]
async fn main() {
    let presets_raw = std::fs::read_to_string(Path::new(REPO_ROOT).join("webui/presets.json")).expect("presets.json");
    let presets_v: Value = serde_json::from_str(&presets_raw).expect("presets.json parse");
    let presets: Vec<Preset> = serde_json::from_value(presets_v["presets"].clone()).expect("presets rows");
    let (tx, _) = broadcast::channel(1024);
    let st = Arc::new(AppState {
        presets,
        runs: Mutex::new(vec![]),
        child: Mutex::new(None),
        events: tx,
        next_id: Mutex::new(0),
    });
    let static_dir = PathBuf::from(REPO_ROOT).join("webui/static");
    let app = Router::new()
        .route("/api/presets", get(get_presets))
        .route("/api/runs", get(list_runs).post(launch))
        .route("/api/runs/:id/stop", post(stop))
        .route("/api/runs/:id/samples", get(samples))
        .route("/api/events", get(events))
        .route("/api/system/metrics", get(system_metrics))
        .route("/api/dataset/media", get(dataset_media))
        .route("/api/caption", get(caption_get).put(caption_put))
        .route("/api/validations", get(validations_get).put(validations_put))
        .route("/api/runs/history", get(runs_history))
        .route("/files/*path", get(file_serve))
        .nest_service("/", tower_http::services::ServeDir::new(static_dir).append_index_html_on_directories(true))
        .with_state(st);
    let addr = "0.0.0.0:8188";
    println!("serenity web trainer on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
