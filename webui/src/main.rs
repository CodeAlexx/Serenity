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
    process::{Child, Command},
    sync::{broadcast, Mutex},
};

mod captioner;
mod board;
// The server bin calls build_merged_config + validate_config_enums (launch
// path); the config_smoke bin exercises the rest of the shared file.
#[allow(dead_code)]
mod config_merge;

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
    /// Per-preset environment for the spawned trainer (wave 3, 2026-07-22).
    /// Needed because several mojodiffusion drivers select variants/data via
    /// env, not argv/config (measured): train_wan22_real reads WAN21_MODEL
    /// (Wan2.1 dispatch happens BEFORE the config is read) and WAN22_DATA_CACHE
    /// (all wan2x arms read the cache dir from env only); train_mageflow_real
    /// reads MAGEFLOW_DATA_CACHE. Empty for every other preset — launches are
    /// byte-identical to the pre-wave-3 behavior when this map is empty.
    #[serde(default)]
    env: HashMap<String, String>,
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
    #[serde(default)]
    message: String, // last error line on failure (finding #7) — surfaced in the UI pill/#msg
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
    // structured trainer banners elevated from log noise (finding #6) — also
    // carried so a reattach/re-adopt shows the last save + resume kind.
    #[serde(default)]
    resume_kind: String, // "" | "full" | "warm"
    #[serde(default)]
    last_save: String, // last [save] wrote / FINAL LoRA path
    #[serde(default)]
    stage: String, // last klein PROG_STAGE phase
}

struct AppState {
    presets: Vec<Preset>,
    runs: Mutex<Vec<RunInfo>>,
    child: Mutex<Option<Child>>, // one run at a time (the 24GB rule)
    // pid of the active run (launched OR re-adopted after a server restart,
    // finding #5). Distinct from `child`: an adopted run has no tokio Child
    // handle, only its pid — stop() and liveness polling key on this.
    active_pid: Mutex<Option<u32>>,
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
/// pub(crate) so the board tailer can feed CLI-run log lines through the SAME parser.
pub(crate) fn parse_progress(line: &str, run: &mut RunInfo) -> bool {
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

/// Elevate the trainer's own banner prints to typed SSE events (finding #6).
/// Mutates the RunInfo's carried fields (resume_kind/last_save/stage) and
/// returns the typed event JSON to broadcast, or None if the line is not a
/// banner (or a PROG_STAGE whose phase is unchanged — deduped so klein's
/// ~10 stage lines/step don't flood the stream). Banner strings verified in
/// train_krea2.mojo, training/trainer_core.mojo, train_klein_real.mojo, and the
/// train_*_real.mojo resume prints.
pub(crate) fn parse_banner(line: &str, run: &mut RunInfo) -> Option<Value> {
    let after = |pat: &str| -> Option<String> {
        line.find(pat).map(|i| line[i + pat.len()..].trim().to_string())
    };
    // resume — WARM (moments restart) vs FULL (moments restored). trainer_core
    // prints "!! WARM RESUME"; each train_*_real prints "[<m>-resume] FULL resume".
    if line.contains("WARM RESUME") {
        run.resume_kind = "warm".into();
        return Some(json!({"type": "resume", "run_id": run.id, "kind": "warm"}));
    }
    if line.contains("-resume]") && line.contains("FULL resume") {
        let path = after(" from ").unwrap_or_default();
        run.resume_kind = "full".into();
        return Some(json!({"type": "resume", "run_id": run.id, "kind": "full", "path": path}));
    }
    // prune — rolling-retention delete ("[prune] removed old ... -> <path>")
    if line.contains("[prune] removed") {
        let path = after("-> ").unwrap_or_default();
        return Some(json!({"type": "prune", "run_id": run.id, "path": path}));
    }
    // save — periodic ("[save] wrote N LoRA pairs -> <p> (+ .a3state)") + final
    if line.contains("FINAL LoRA") {
        let path = after("-> ").unwrap_or_default();
        run.last_save = path.clone();
        return Some(json!({"type": "save", "run_id": run.id, "path": path, "final": true}));
    }
    if line.contains("[save] wrote") {
        let mut path = after("-> ").unwrap_or_default();
        if let Some(i) = path.find(" (+") {
            path.truncate(i); // drop the " (+ .a3state)" suffix
        }
        let path = path.trim().to_string();
        run.last_save = path.clone();
        return Some(json!({"type": "save", "run_id": run.id, "path": path}));
    }
    // save — OT-family periodic shape: "[<M>-lora] save_state step= N  path= <p>"
    // (chroma/sd35/anima/... use ot_step_lora_path; measured: chroma emits no
    // "[save] wrote" — this shape is their only per-save line.)
    if line.contains("save_state step=") && line.contains("path=") {
        let mut path = after("path=").unwrap_or_default();
        if let Some(i) = path.find(".state.safetensors") {
            path.truncate(i); // badge shows the checkpoint, not the sidecar
        }
        let path = path.trim().to_string();
        run.last_save = path.clone();
        return Some(json!({"type": "save", "run_id": run.id, "path": path}));
    }
    // klein PROG_STAGE step=<k> ... phase=<phase> — emit only on phase change
    if line.contains("PROG_STAGE step=") {
        let phase = line
            .find("phase=")
            .map(|i| line[i + 6..].split_whitespace().next().unwrap_or("").to_string())
            .unwrap_or_default();
        if phase.is_empty() || phase == run.stage {
            run.stage = phase;
            return None;
        }
        run.stage = phase.clone();
        return Some(json!({"type": "stage", "run_id": run.id, "phase": phase}));
    }
    None
}

// ── finding #5: persist the active run so a server restart re-adopts it ───────

/// One-active-run metadata mirrored to disk (the 24GB rule = at most one). On
/// boot we read it, and if the pid is still alive we re-adopt the live trainer;
/// if the pid is gone we finalize it from the log tail. Written at launch,
/// cleared at finalize/stop.
#[derive(Serialize, Deserialize, Clone)]
struct ActiveRun {
    id: u64,
    preset_id: String,
    backend: String,
    binary: String,
    workspace_dir: String,
    log_path: String,
    config_path: String,
    pid: u32,
    start_time: f64,
    total_steps: u64,
}

fn active_run_path() -> String {
    format!("{REPO_ROOT}/webui/active_run.json")
}

fn write_active_run(a: &ActiveRun) {
    let _ = std::fs::write(active_run_path(), serde_json::to_string_pretty(a).unwrap_or_default());
}

fn read_active_run() -> Option<ActiveRun> {
    std::fs::read_to_string(active_run_path()).ok().and_then(|s| serde_json::from_str(&s).ok())
}

fn clear_active_run() {
    let _ = std::fs::remove_file(active_run_path());
}

/// pid liveness: /proc/<pid> exists AND (if readable) its cmdline still names
/// the trainer binary — guards against adopting a reused pid after a reboot.
fn pid_is_trainer(pid: u32, binary: &str) -> bool {
    if !Path::new(&format!("/proc/{pid}")).exists() {
        return false;
    }
    let base = Path::new(binary).file_name().and_then(|b| b.to_str()).unwrap_or("");
    match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(raw) => {
            let cmd = String::from_utf8_lossy(&raw);
            base.is_empty() || cmd.contains(base) || cmd.contains("serenity_")
        }
        Err(_) => true, // alive but cmdline unreadable — treat as adoptable
    }
}

/// Scan the tail of a failed run's log for the last error line (finding #7,
/// shared by the launch tail loop and the re-adopt loop).
fn scan_last_error(log_path: &str) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let mut msg = String::new();
    if let Ok(mut f) = std::fs::File::open(log_path) {
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        let _ = f.seek(SeekFrom::Start(len.saturating_sub(65536)));
        let mut raw = Vec::new();
        if f.read_to_end(&mut raw).is_ok() {
            for line in String::from_utf8_lossy(&raw).lines() {
                if line.contains("Error") || line.contains("Unhandled exception") {
                    msg = line.trim().to_string();
                }
            }
        }
    }
    msg
}

/// Reconstruct the latest progress + banner state from an existing log's tail,
/// so a re-adopted run shows its current step/loss/last-save immediately.
fn recover_progress(log_path: &str, info: &mut RunInfo) {
    use std::io::{Read, Seek, SeekFrom};
    if let Ok(mut f) = std::fs::File::open(log_path) {
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        let _ = f.seek(SeekFrom::Start(len.saturating_sub(65536)));
        let mut raw = Vec::new();
        if f.read_to_end(&mut raw).is_ok() {
            for line in String::from_utf8_lossy(&raw).lines() {
                let l = line.trim();
                if l.is_empty() {
                    continue;
                }
                let _ = parse_progress(l, info);
                let _ = parse_banner(l, info);
            }
        }
    }
}

/// A clean finish for every wired trainer prints the FINAL LoRA banner. Used to
/// classify a run whose pid is already gone at boot (exited vs failed).
fn log_indicates_success(log_path: &str) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    if let Ok(mut f) = std::fs::File::open(log_path) {
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        let _ = f.seek(SeekFrom::Start(len.saturating_sub(131072)));
        let mut raw = Vec::new();
        if f.read_to_end(&mut raw).is_ok() {
            let s = String::from_utf8_lossy(&raw);
            // krea2-family prints "FINAL LoRA"; the OT-family drivers print
            // "RESULT: REAL run OK" (measured: chroma run misclassified failed).
            return s.contains("FINAL LoRA") || s.contains("RESULT: REAL run OK");
        }
    }
    false
}

/// Terminal bookkeeping shared by the launch tail loop and the re-adopt loop:
/// clear the active-run file, set status (guarding a prior stop/failed), record
/// history + board, and broadcast the status event.
async fn finalize_run(st: &Arc<AppState>, id: u64, log_path: &str, success: bool) {
    let fail_msg = if success { String::new() } else { scan_last_error(log_path) };
    clear_active_run();
    *st.active_pid.lock().await = None;
    let mut runs = st.runs.lock().await;
    if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
        if r.status == "running" {
            r.status = if success { "exited".into() } else { "failed".into() };
        }
        if !fail_msg.is_empty() {
            r.message = fail_msg;
        }
        append_history(r);
        board::run_ended(&r.workspace_dir, &r.status);
        let _ = st.events.send(json!({"type": "status", "run": &*r}).to_string());
    }
}

/// Re-adopt loop for a run inherited across a server restart: no tokio Child, so
/// liveness is polled via /proc/<pid>. Tails NEW log lines from `offset` for
/// parse + SSE, and finalizes when the pid disappears.
fn spawn_adopt_loop(st: Arc<AppState>, id: u64, log_path: String, pid: u32, mut offset: u64) {
    tokio::spawn(async move {
        let mut pending = String::new();
        loop {
            if let Ok(mut f) = std::fs::File::open(&log_path) {
                use std::io::{Read, Seek, SeekFrom};
                let len = f.metadata().map(|m| m.len()).unwrap_or(0);
                if len > offset {
                    let _ = f.seek(SeekFrom::Start(offset));
                    let mut buf = String::new();
                    if f.read_to_string(&mut buf).is_ok() {
                        offset = len;
                        pending.push_str(&buf);
                        while let Some(nl) = pending.find('\n') {
                            let line: String = pending.drain(..=nl).collect();
                            let line = line.trim_end();
                            if line.is_empty() {
                                continue;
                            }
                            let mut runs = st.runs.lock().await;
                            if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
                                let parsed = parse_progress(line, r);
                                if parsed {
                                    board::ingest(&r.workspace_dir, r.step, r.epoch, r.loss, r.grad_norm, r.s_per_step, r.total_steps);
                                    let _ = st.events.send(json!({"type": "progress", "run": &*r}).to_string());
                                } else {
                                    let _ = st.events.send(json!({"type": "log", "run_id": id, "line": line}).to_string());
                                    if let Some(bev) = parse_banner(line, r) {
                                        let _ = st.events.send(bev.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !Path::new(&format!("/proc/{pid}")).exists() {
                tokio::time::sleep(Duration::from_millis(300)).await;
                let success = log_indicates_success(&log_path);
                finalize_run(&st, id, &log_path, success).await;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });
}

/// On boot, re-adopt or finalize the persisted active run (finding #5).
async fn adopt_persisted_run(st: &Arc<AppState>) {
    let Some(a) = read_active_run() else { return };
    let mut info = RunInfo {
        id: a.id,
        preset_id: a.preset_id.clone(),
        backend: a.backend.clone(),
        workspace_dir: a.workspace_dir.clone(),
        log_path: a.log_path.clone(),
        status: "running".into(),
        message: String::new(),
        pid: Some(a.pid),
        step: 0,
        total_steps: a.total_steps,
        epoch: 0,
        total_epochs: 0,
        loss: 0.0,
        grad_norm: 0.0,
        s_per_step: 0.0,
        eta: String::new(),
        resume_kind: String::new(),
        last_save: String::new(),
        stage: String::new(),
    };
    recover_progress(&a.log_path, &mut info);
    // keep new run ids monotonic past the adopted one
    {
        let mut n = st.next_id.lock().await;
        if a.id > *n {
            *n = a.id;
        }
    }
    if pid_is_trainer(a.pid, &a.binary) {
        info.message = format!("▶ re-adopted after restart: tail -f {}", a.log_path);
        let offset = std::fs::metadata(&a.log_path).map(|m| m.len()).unwrap_or(0);
        st.runs.lock().await.push(info);
        *st.active_pid.lock().await = Some(a.pid);
        eprintln!("[adopt] re-adopted live run {} (pid {}) from {}", a.id, a.pid, a.log_path);
        spawn_adopt_loop(st.clone(), a.id, a.log_path.clone(), a.pid, offset);
    } else {
        // pid gone while we were down — classify from the log and record it
        let success = log_indicates_success(&a.log_path);
        info.status = if success { "exited".into() } else { "failed".into() };
        if !success {
            info.message = scan_last_error(&a.log_path);
        }
        append_history(&info);
        board::run_ended(&info.workspace_dir, &info.status);
        st.runs.lock().await.push(info);
        clear_active_run();
        eprintln!("[adopt] persisted run {} pid {} gone -> {}", a.id, a.pid, if success { "exited" } else { "failed" });
    }
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
    // Single source of truth (config_merge::build_merged_config) so the offline
    // webui-config-smoke merges IDENTICALLY (finding #3). Err = base template
    // named but missing/unreadable -> 422 (unchanged behavior). The sampling
    // strip for sd35|hidream|ideogram4 (finding #8) lives inside the shared fn.
    let run_name = req.run_name.clone().unwrap_or_else(|| p.run_name.clone());
    let workspace = format!("/home/alex/mojodiffusion/output/{run_name}");
    let (cfg, notes) = match config_merge::build_merged_config(
        REPO_ROOT, &p.base_config, &p.recipe, &p.backend, &run_name, &req.overrides,
    ) {
        Ok(v) => v,
        Err(e) => return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": e}))),
    };
    // P1 fix #4: enum-validate on the LAUNCH path too (previously only the
    // offline config_smoke ran this), so a reader-rejected enum is a 422 in
    // the UI instead of a runtime fail-loud inside the spawned child.
    let enum_errs = config_merge::validate_config_enums(&cfg);
    if !enum_errs.is_empty() {
        return (StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": format!("config rejected by trainer enum validation: {}", enum_errs.join("; "))})));
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
    // ---- P2 (UI_WIRING_CAMPAIGN 2026-07-22): ideogram4 levers delivery ----
    // The ideogram4 shape is all-argv; before this the merged config was written
    // to disk but NEVER delivered — every non-argv lever silently discarded.
    // Fix: extract the LEVER subset the runner's argv-11 path actually consumes
    // (Ideogram4LiveTrainer.mojo:158-287 reads argv 11 via serenitymojo
    // read_model_config; Ideogram4LoRATrainer.mojo lcfg usage: loss/optimizer/
    // EMA/caption-dropout-fallback/batch_size/adapter_algo+LoKr knobs) into a
    // levers JSON in the workspace and pass its path as argv 11. Recipe scalars
    // (lr/rank/alpha/steps/save) stay argv-owned per the seam contract. With NO
    // lever key present argv 11 stays "-" (trainer skip sentinel, C13) so
    // default launches remain byte-identical to the pre-fix line.
    let mut ideogram4_levers_arg = String::from("-");
    if p.argv_shape == "ideogram4" {
        const IDEOGRAM4_LEVER_KEYS: &[&str] = &[
            // loss levers (train_config_reader.mojo:1123-1133)
            "loss_fn", "huber_delta", "smooth_l1_beta", "min_snr_gamma_flow",
            // optimizer levers: nested object (reader _parse_optimizer) + warmup
            // aliases (reader l.1100-1105)
            "optimizer", "optimizer_warmup_steps", "lr_warmup_steps",
            "learning_rate_warmup_steps",
            // EMA levers (reader l.1175-1193)
            "ema", "ema_enabled", "ema_inv_gamma", "ema_power",
            "ema_update_after_step", "ema_min_decay", "ema_max_decay",
            "ema_decay", "ema_update_step_interval",
            // caption-dropout fallback (argv 10 wins when > 0; reader l.1157)
            "caption_dropout_prob",
            // TRUE batch-2 selector (LiveTrainer.mojo:275-278)
            "batch_size",
            // LyCORIS carrier dispatch + LoKr knobs (reader l.1211-1246)
            "network_algorithm", "algo", "adapter_algo", "lokr_targets",
            "lokr_factor", "lokr_decompose_both", "lokr_full_matrix",
            "init_lokr_norm",
        ];
        let mut levers = serde_json::Map::new();
        if let Value::Object(m) = &cfg {
            for k in IDEOGRAM4_LEVER_KEYS {
                if let Some(v) = m.get(*k) {
                    levers.insert((*k).to_string(), v.clone());
                }
            }
        }
        if !levers.is_empty() {
            // model_type documents the emitting seam (reader: cfg.name only)
            levers.insert("model_type".into(), json!("ideogram4"));
            let levers_path = format!("{workspace}/web_levers.json");
            if std::fs::write(&levers_path, serde_json::to_string_pretty(&Value::Object(levers)).unwrap()).is_err() {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "cannot write levers config"})));
            }
            ideogram4_levers_arg = levers_path;
        }
    }
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
        // argv 11 = levers JSON path when any lever key is in the merged config
        // (P2 delivery fix above), else "-" (defaults stay byte-identical, C13).
        // argv 13/14 sampler steps+cfg and argv 18 resolution come from the
        // merged config (preset recipe defaults 20 / 4.5 / 512, overridable).
        // argv 16 resume "-", argv 17 prompt "-" (no inline sampling in unit 2).
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
            // UI emit renamed caption_dropout -> caption_dropout_prob (the
            // shared reader's key); argv10 reads the same renamed key.
            format!("{}", getf("caption_dropout_prob", 0.0)),
            ideogram4_levers_arg.clone(),
            format!("{}", getu("sample_every", 0)),
            format!("{}", getu("sample_steps", 20)),
            format!("{}", getf("sample_cfg", 4.5)),
            format!("{}", getu("seed", 42)),
            "-".into(),
            "-".into(),
            format!("{}", getu("resolution", 512)),
        ],
        // shape 5 (wave 3): LTX-2 AV production trainer (mojodiffusion
        // training/train_ltx2_av.mojo). Flag-based CLI (LTX2TrainerConfig::
        // from_args) — NOT the positional config_runner shape; positional args
        // are silently IGNORED by its parser, so config_runner would launch it
        // with pure defaults. --config carries the merged lever JSON; the
        // recipe scalars ride their own flags (ltx2 argv wins for its fields).
        // --ltx2_mode video is REQUIRED (the struct default is MODE_AV, which
        // fail-louds without --lora_target_preset audio).
        "ltx2" => {
            let mut v = vec![
                "--config".into(), cfg_path.clone(),
                "--ltx2_mode".into(), "video".into(),
                "--geometry".into(),
                cfg.get("geometry").and_then(|g| g.as_str()).unwrap_or("image512").to_string(),
                "--dataset_cache_dir".into(), cache.clone(),
                "--output_dir".into(), workspace.clone(),
                "--max_steps".into(), steps.to_string(),
                "--lora_rank".into(), format!("{}", getu("lora_rank", 32)),
                "--lora_alpha".into(), format!("{}", getf("lora_alpha", 32.0)),
                "--learning_rate".into(), format!("{}", getf("learning_rate", 1e-4)),
                "--save_every".into(), format!("{}", getu("save_every", 0)),
                "--sample_every".into(), format!("{}", getu("sample_every", 0)),
                "--seed".into(), format!("{}", getu("seed", 42)),
                // 16GB refit (measured 2026-07-22): the trainer's non-capture
                // default is 42 fp8-resident blocks (~17GB, sized for the 24GB
                // box) -> generic CUDA OOM in the load phase on the 5080.
                // Residency changes NO math (anchor-identical); preset-tunable.
                "--resident_blocks".into(), format!("{}", getu("resident_blocks", 12)),
            ];
            if !p.checkpoint.is_empty() {
                v.push("--ltx2_checkpoint".into());
                v.push(p.checkpoint.clone());
            }
            v
        }
        other => return (StatusCode::NOT_IMPLEMENTED, Json(json!({"error": format!("argv shape {other} not wired")}))),
    };
    // finding #1: resume argv order is PER-BACKEND, not per argv-shape. The
    // config_runner shape is shared by many backends whose trainers parse resume
    // args in different slots (or not at all), so key the resume append on the
    // backend and use the verified slot order for each. Backends with NO resume
    // parse must be REFUSED (a 400) rather than silently launched fresh.
    //   krea2  train_krea2.mojo:3062-3069     -> [.., resume_path, start_step]
    //   klein  train_klein_real.mojo:1121-1133 -> [.., start_step, resume_path]
    //   zimage train_zimage_real.mojo:3161-3168 -> [.., start_step, resume_path]
    //   sd35   train_sd35_real.mojo:801-807     -> [.., resume_path]  (raises on a
    //          3rd extra arg; start step is auto-probed from the checkpoint)
    if let Some(state) = &req.resume_state {
        if !state.ends_with(".state") && !state.contains(".state.") {
            return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "resume_state must be the .state file path (MJ-1077: the PEFT path silently warm-resumes)"})));
        }
        let start_step = req.start_step.unwrap_or(0).to_string();
        match p.backend.as_str() {
            "krea2" => {
                args.push(state.clone());
                args.push(start_step);
            }
            "klein" | "zimage" => {
                args.push(start_step);
                args.push(state.clone());
            }
            "sd35" => {
                // sd35 has no start_step slot; a 3rd extra arg makes it fail loud.
                args.push(state.clone());
            }
            other => {
                return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("backend {other} does not support resume")})));
            }
        }
    }
    if req.dry_run {
        return (StatusCode::OK, Json(json!({"dry_run": true, "binary": p.binary, "args": args, "env": p.env, "config_written": cfg_path, "config": cfg, "notes": notes.clone()})));
    }
    let log_path = format!("{workspace}/train_web.log");
    let log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("log create: {e}")}))),
    };
    // DECOUPLED OUTPUT (2026-07-05 incident): the child writes its log FILE
    // directly — no pipe to this server. A server restart/crash can no longer
    // break the child (a piped child dies on SIGPIPE at its next print once
    // the server is gone; KillMode=process alone doesn't cover that). The
    // supervisor TAILS the file for parsing/SSE instead. stdbuf -oL keeps
    // lines flushing promptly to the file for live tailing.
    let mut cmd = Command::new("stdbuf");
    cmd.arg("-oL")
        .arg("-eL")
        .arg(Path::new(REPO_ROOT).join(&p.binary))
        .args(&args)
        .current_dir(REPO_ROOT)
        .env("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "true")
        .env("LD_LIBRARY_PATH", CUDA_LD)
        // per-preset trainer env (wave 3): variant/data selection some drivers
        // read from env only (WAN21_MODEL / WAN22_DATA_CACHE / MAGEFLOW_DATA_CACHE)
        .envs(&p.env)
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
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
        message: String::new(),
        pid,
        step: 0,
        total_steps: steps,
        epoch: 0,
        total_epochs: 0,
        loss: 0.0,
        grad_norm: 0.0,
        s_per_step: 0.0,
        eta: String::new(),
        resume_kind: String::new(),
        last_save: String::new(),
        stage: String::new(),
    };
    st.runs.lock().await.push(info.clone());
    board::run_started(&workspace, &p.id, steps, &cfg);
    // TAIL the child's log file -> parse + broadcast SSE (no pipe: see spawn note)
    *st.child.lock().await = Some(child);
    // finding #5: mirror the active run to disk + track its pid so a server
    // restart re-adopts (adopt_persisted_run) instead of orphaning it.
    if let Some(pidv) = pid {
        *st.active_pid.lock().await = Some(pidv);
        write_active_run(&ActiveRun {
            id,
            preset_id: p.id.clone(),
            backend: p.backend.clone(),
            binary: p.binary.clone(),
            workspace_dir: workspace.clone(),
            log_path: log_path.clone(),
            config_path: cfg_path.clone(),
            pid: pidv,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            total_steps: steps,
        });
    }
    let stc = st.clone();
    tokio::spawn(async move {
        let mut offset: u64 = 0;
        let mut pending = String::new();
        loop {
            // read any new bytes
            if let Ok(mut f) = std::fs::File::open(&log_path) {
                use std::io::{Read, Seek, SeekFrom};
                let len = f.metadata().map(|m| m.len()).unwrap_or(0);
                if len > offset {
                    let _ = f.seek(SeekFrom::Start(offset));
                    let mut buf = String::new();
                    if f.read_to_string(&mut buf).is_ok() {
                        offset = len;
                        pending.push_str(&buf);
                        while let Some(nl) = pending.find('\n') {
                            let line: String = pending.drain(..=nl).collect();
                            let line = line.trim_end();
                            if line.is_empty() { continue; }
                            let mut runs = stc.runs.lock().await;
                            if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
                                let parsed = parse_progress(line, r);
                                if parsed {
                                    board::ingest(&r.workspace_dir, r.step, r.epoch, r.loss, r.grad_norm, r.s_per_step, r.total_steps);
                                    let _ = stc.events.send(json!({"type": "progress", "run": &*r}).to_string());
                                } else {
                                    // keep the raw line in the log pane, THEN elevate
                                    // recognized banners to a typed event (finding #6).
                                    let _ = stc.events.send(json!({"type": "log", "run_id": id, "line": line}).to_string());
                                    if let Some(bev) = parse_banner(line, r) {
                                        let _ = stc.events.send(bev.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // child exited AND log drained -> finalize
            let exited = {
                let mut ch = stc.child.lock().await;
                match ch.as_mut() {
                    Some(c) => match c.try_wait() {
                        Ok(Some(status)) => {
                            *ch = None;
                            Some(status.success())
                        }
                        _ => None,
                    },
                    None => Some(false), // stopped externally
                }
            };
            if let Some(success) = exited {
                // one final drain pass for the tail of the log
                tokio::time::sleep(Duration::from_millis(300)).await;
                if let Ok(mut f) = std::fs::File::open(&log_path) {
                    use std::io::{Read, Seek, SeekFrom};
                    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
                    if len > offset {
                        let _ = f.seek(SeekFrom::Start(offset));
                        let mut buf = String::new();
                        let _ = f.read_to_string(&mut buf);
                        for line in buf.lines().filter(|l| !l.trim().is_empty()) {
                            let mut runs = stc.runs.lock().await;
                            if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
                                if parse_progress(line, r) {
                                    board::ingest(&r.workspace_dir, r.step, r.epoch, r.loss, r.grad_norm, r.s_per_step, r.total_steps);
                                }
                                let _ = stc.events.send(json!({"type": "log", "run_id": id, "line": line}).to_string());
                                // elevate the end-of-run banners (FINAL LoRA / prune)
                                if let Some(bev) = parse_banner(line, r) {
                                    let _ = stc.events.send(bev.to_string());
                                }
                            }
                        }
                    }
                }
                // finding #7: on non-zero exit, surface the actual error line
                // (finalize_run scans the tail); clears active_run + active_pid.
                finalize_run(&stc, id, &log_path, success).await;
                break;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
    });
    (StatusCode::OK, Json(json!({"run_id": id, "workspace": workspace, "log": info.log_path, "tail": format!("tail -f {}", info.log_path), "notes": notes})))
}

async fn stop(State(st): State<Arc<AppState>>, AxPath(id): AxPath<u64>) -> (StatusCode, Json<Value>) {
    // Kill by pid (covers BOTH a launched child and a re-adopted run that has no
    // tokio Child handle, finding #5). SIGTERM, grace, then reap the child.
    let target = *st.active_pid.lock().await;
    if let Some(pid) = target {
        unsafe { libc_kill(pid as i32, 15) };
    }
    {
        let mut ch = st.child.lock().await;
        if let Some(c) = ch.as_mut() {
            if let Some(pid) = c.id() {
                unsafe { libc_kill(pid as i32, 15) };
            }
        }
        if target.is_some() || ch.is_some() {
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        if let Some(c) = ch.as_mut() {
            let _ = c.start_kill();
        }
        *ch = None;
    }
    if let Some(pid) = target {
        unsafe { libc_kill(pid as i32, 9) }; // adopted run: SIGKILL fallback
    }
    *st.active_pid.lock().await = None;
    clear_active_run();
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
    if !std::path::Path::new(&q.path).is_dir() {
        return Json(json!({"error": format!("folder not found: {}", q.path)}));
    }
    // Recursive to depth 2 (folder + one subdir level): users point at a dataset
    // PARENT (e.g. 1_giger/ with images in gigerver3/) and expect to see them.
    let mut files: Vec<std::path::PathBuf> = vec![];
    if let Ok(rd) = std::fs::read_dir(&q.path) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with('.')).unwrap_or(true) {
                    continue;
                }
                if let Ok(rd2) = std::fs::read_dir(&p) {
                    files.extend(rd2.flatten().map(|e2| e2.path()).filter(|p2| p2.is_file()));
                }
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    let mut items = vec![];
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

/// Full UI state persistence (theme + every form field) — server-owned so it
/// survives any browser/machine, unlike localStorage.
fn ui_state_path() -> String {
    format!("{REPO_ROOT}/webui/ui_state.json")
}

async fn ui_state_get() -> Json<Value> {
    Json(std::fs::read_to_string(ui_state_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(|| json!({})))
}

async fn ui_state_put(Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    match std::fs::write(ui_state_path(), serde_json::to_string_pretty(&body).unwrap_or_default()) {
        Ok(_) => (StatusCode::OK, Json(json!({"saved": true}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    }
}

/// Named saved configs (OneTrainer-style Save/Load config) — one JSON per name
/// under webui/saved_configs/. Body: {preset_id, run_name, fields:{id:value}}.
fn saved_configs_dir() -> String {
    format!("{REPO_ROOT}/webui/saved_configs")
}

fn config_name_ok(name: &str) -> bool {
    !name.is_empty() && name.len() <= 80
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ' ' || c == '.')
}

async fn configs_list() -> Json<Value> {
    let mut names: Vec<String> = std::fs::read_dir(saved_configs_dir()).map(|rd| rd
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .filter_map(|n| n.strip_suffix(".json").map(String::from))
        .collect()).unwrap_or_default();
    names.sort();
    Json(json!({"configs": names}))
}

async fn config_get(axum::extract::Path(name): axum::extract::Path<String>) -> (StatusCode, Json<Value>) {
    if !config_name_ok(&name) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad config name"})));
    }
    let p = format!("{}/{}.json", saved_configs_dir(), name);
    match std::fs::read_to_string(&p).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
        Some(v) => (StatusCode::OK, Json(v)),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": format!("config '{name}' not found")}))),
    }
}

async fn config_put(axum::extract::Path(name): axum::extract::Path<String>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    if !config_name_ok(&name) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad config name"})));
    }
    let dir = saved_configs_dir();
    let _ = std::fs::create_dir_all(&dir);
    match std::fs::write(format!("{dir}/{name}.json"), serde_json::to_string_pretty(&body).unwrap_or_default()) {
        Ok(_) => (StatusCode::OK, Json(json!({"saved": name}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    }
}

/// Server-side filesystem browser for path pickers. Bounded to /home/alex and
/// /mnt; returns dirs + files (names only) for one directory.
#[derive(serde::Deserialize)]
struct FsQuery { path: Option<String> }

async fn fs_list(axum::extract::Query(q): axum::extract::Query<FsQuery>) -> (StatusCode, Json<Value>) {
    let path = q.path.unwrap_or_else(|| "/home/alex".into());
    let canon = std::fs::canonicalize(&path).unwrap_or_else(|_| std::path::PathBuf::from("/home/alex"));
    let cs = canon.to_string_lossy().to_string();
    if !(cs == "/home/alex" || cs.starts_with("/home/alex/") || cs == "/mnt" || cs.starts_with("/mnt/")) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "path outside /home/alex and /mnt"})));
    }
    let mut dirs: Vec<String> = vec![];
    let mut files: Vec<String> = vec![];
    match std::fs::read_dir(&canon) {
        Ok(rd) => {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with('.') { continue; }
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) { dirs.push(name); } else { files.push(name); }
            }
        }
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))),
    }
    dirs.sort(); files.sort();
    let parent = canon.parent().map(|p| p.to_string_lossy().to_string());
    (StatusCode::OK, Json(json!({"path": cs, "parent": parent, "dirs": dirs, "files": files})))
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
        active_pid: Mutex::new(None),
        events: tx,
        next_id: Mutex::new(0),
    });
    board::init();
    // finding #5: re-adopt (or finalize) a run inherited across a server restart
    // BEFORE serving, so /api/runs reflects reality the moment the UI reconnects.
    adopt_persisted_run(&st).await;
    let static_dir = PathBuf::from(REPO_ROOT).join("webui/static");
    let app = Router::new()
        .merge(board::router())
        .route("/api/presets", get(get_presets))
        .route("/api/runs", get(list_runs).post(launch))
        .route("/api/runs/:id/stop", post(stop))
        .route("/api/runs/:id/samples", get(samples))
        .route("/api/events", get(events))
        .route("/api/system/metrics", get(system_metrics))
        .route("/api/dataset/media", get(dataset_media))
        .route("/api/caption", get(caption_get).put(caption_put))
        .route("/api/captioner/run", post(captioner::run))
        .route("/api/captioner/status", get(captioner::status))
        .route("/api/captioner/abort", post(captioner::abort))
        .route("/api/validations", get(validations_get).put(validations_put))
        .route("/api/runs/history", get(runs_history))
        .route("/api/ui/state", get(ui_state_get).put(ui_state_put))
        .route("/api/configs", get(configs_list))
        .route("/api/configs/:name", get(config_get).put(config_put))
        .route("/api/fs", get(fs_list))
        .route("/files/*path", get(file_serve))
        .nest_service("/", tower_http::services::ServeDir::new(static_dir).append_index_html_on_directories(true))
        .with_state(st);
    let addr = "0.0.0.0:8188";
    println!("serenity web trainer on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    // Offline coverage for the wave-2 pure logic: the banner parser (finding #6)
    // and the re-adopt predicates (finding #5). No server, GPU, or board.db.
    use super::*;

    fn blank_run() -> RunInfo {
        RunInfo {
            id: 7,
            preset_id: String::new(),
            backend: String::new(),
            workspace_dir: String::new(),
            log_path: String::new(),
            status: "running".into(),
            message: String::new(),
            pid: None,
            step: 0,
            total_steps: 0,
            epoch: 0,
            total_epochs: 0,
            loss: 0.0,
            grad_norm: 0.0,
            s_per_step: 0.0,
            eta: String::new(),
            resume_kind: String::new(),
            last_save: String::new(),
            stage: String::new(),
        }
    }

    #[test]
    fn banner_full_resume() {
        // train_krea2.mojo:2264 exact print shape
        let mut r = blank_run();
        let ev = parse_banner(
            "[krea2-resume] FULL resume (A/B + AdamW moments) from /out/krea2/ckpt_1500.state",
            &mut r,
        )
        .expect("full resume is a banner");
        assert_eq!(ev["type"], "resume");
        assert_eq!(ev["kind"], "full");
        assert_eq!(ev["path"], "/out/krea2/ckpt_1500.state");
        assert_eq!(r.resume_kind, "full");
    }

    #[test]
    fn banner_warm_resume_not_misread_as_full() {
        // trainer_core.mojo:187 — also contains "-resume]" but must be WARM.
        let mut r = blank_run();
        let ev = parse_banner(
            "  [krea2-resume] !! WARM RESUME — AdamW moments RESTART at zero !!",
            &mut r,
        )
        .expect("warm resume is a banner");
        assert_eq!(ev["kind"], "warm");
        assert_eq!(r.resume_kind, "warm");
    }

    #[test]
    fn banner_save_strips_a3state_suffix() {
        // train_krea2.mojo:3869 "[save] wrote N LoRA pairs -> <p> (+ .a3state)"
        let mut r = blank_run();
        let ev = parse_banner(
            "  [save] wrote 128 LoRA pairs -> /out/krea2/krea2_step1500.safetensors (+ .a3state)",
            &mut r,
        )
        .expect("save is a banner");
        assert_eq!(ev["type"], "save");
        assert_eq!(ev["path"], "/out/krea2/krea2_step1500.safetensors");
        assert_eq!(r.last_save, "/out/krea2/krea2_step1500.safetensors");
    }

    #[test]
    fn banner_final_lora() {
        // train_krea2.mojo:4526 "[save] FINAL LoRA: N pairs ( M tensors) -> <p>"
        let mut r = blank_run();
        let ev = parse_banner(
            "[save] FINAL LoRA: 128 pairs ( 256 tensors) -> /out/krea2/krea2_final.safetensors",
            &mut r,
        )
        .expect("final is a banner");
        assert_eq!(ev["type"], "save");
        assert_eq!(ev["final"], true);
        assert_eq!(ev["path"], "/out/krea2/krea2_final.safetensors");
    }

    #[test]
    fn banner_prune() {
        // trainer_core.mojo:130
        let mut r = blank_run();
        let ev = parse_banner("  [prune] removed old checkpoint -> /out/krea2/ckpt_1000.safetensors", &mut r)
            .expect("prune is a banner");
        assert_eq!(ev["type"], "prune");
        assert_eq!(ev["path"], "/out/krea2/ckpt_1000.safetensors");
    }

    #[test]
    fn banner_stage_dedupes_on_unchanged_phase() {
        // train_klein_real.mojo PROG_STAGE — emit once per phase change only.
        let mut r = blank_run();
        let first = parse_banner("PROG_STAGE step= 12  total= 2000  phase=optim", &mut r);
        assert!(first.is_some());
        assert_eq!(first.unwrap()["phase"], "optim");
        // same phase next line -> no event (dedupe), state unchanged
        let second = parse_banner("PROG_STAGE step= 13  total= 2000  phase=optim", &mut r);
        assert!(second.is_none());
        assert_eq!(r.stage, "optim");
        // phase change -> event again
        let third = parse_banner("PROG_STAGE step= 13  total= 2000  phase=backward", &mut r);
        assert_eq!(third.unwrap()["phase"], "backward");
    }

    #[test]
    fn banner_ignores_ordinary_and_progress_lines() {
        let mut r = blank_run();
        assert!(parse_banner("loading base weights ...", &mut r).is_none());
        assert!(parse_banner("[X] step 12/2000 | loss 0.5 | 2.1s/step", &mut r).is_none());
    }

    #[test]
    fn recover_progress_from_log_tail() {
        let dir = std::env::temp_dir();
        let log = dir.join(format!("cs_recover_{}.log", std::process::id()));
        std::fs::write(
            &log,
            "boot\n[X] step 1500/2000 | epoch 3/4 | loss 0.4211 | grad_norm 0.12 | 2.0s/step | ETA 0:10:00\n  [save] wrote 64 LoRA pairs -> /out/x_step1500.safetensors\n",
        )
        .unwrap();
        let mut r = blank_run();
        recover_progress(log.to_str().unwrap(), &mut r);
        assert_eq!(r.step, 1500);
        assert_eq!(r.total_steps, 2000);
        assert!((r.loss - 0.4211).abs() < 1e-6);
        assert_eq!(r.last_save, "/out/x_step1500.safetensors");
        let _ = std::fs::remove_file(&log);
    }

    #[test]
    fn log_success_marker() {
        let dir = std::env::temp_dir();
        let ok = dir.join(format!("cs_ok_{}.log", std::process::id()));
        let bad = dir.join(format!("cs_bad_{}.log", std::process::id()));
        std::fs::write(&ok, "step 2000/2000\n[save] FINAL LoRA: 64 pairs ( 128 tensors) -> /out/f.safetensors\n").unwrap();
        std::fs::write(&bad, "step 812/2000\nUnhandled exception: CUDA OOM\n").unwrap();
        assert!(log_indicates_success(ok.to_str().unwrap()));
        assert!(!log_indicates_success(bad.to_str().unwrap()));
        let _ = std::fs::remove_file(&ok);
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn pid_liveness_predicate() {
        // a pid that cannot be alive
        assert!(!pid_is_trainer(4_000_000_000, "serenity_krea2_live_trainer"));
        // a real live child whose cmdline names the "binary" basename
        let child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id();
        std::thread::sleep(Duration::from_millis(150)); // let fork->exec populate /proc/<pid>/cmdline
        assert!(pid_is_trainer(pid, "/usr/bin/sleep"));
        unsafe { libc_kill(pid as i32, 9) };
    }
}
