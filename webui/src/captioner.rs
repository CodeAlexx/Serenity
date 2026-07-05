// Folder captioner endpoints for the serenity web trainer.
//
// Spawns webui/captioner.py (ai-toolkit venv) over an image folder, pumps its
// CAPJSON progress lines into a polled status object, and writes .txt sidecars
// next to each image. Self-contained module state (OnceLock) so main.rs only
// grows by `mod captioner;` + three route lines — the training path and its
// AppState are untouched. Mutual exclusion with training is via the shared GPU:
// a compute process >1GB makes either side refuse.

use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

const PY: &str = "/home/alex/ai-toolkit/venv/bin/python";
const SCRIPT: &str = "/home/alex/serenity-trainer/webui/captioner.py";
const STDERR_LOG: &str = "/home/alex/serenity-trainer/webui/captioner_last.log";

#[derive(Default, Clone, Serialize)]
struct Result {
    file: String,
    caption: String,
    sidecar: String,
}

#[derive(Default, Clone, Serialize)]
struct CapStatus {
    running: bool,
    finished: bool,
    aborted: bool,
    folder: String,
    model: String,
    total: u64,
    found: u64,
    done: u64,
    current_file: String,
    last_caption: String,
    error: String, // last non-fatal per-file error
    fatal: String, // fatal error (bad folder, model load crash)
    results: Vec<Result>,
}

struct CapState {
    status: Mutex<CapStatus>,
    child: tokio::sync::Mutex<Option<Child>>,
}

fn state() -> &'static CapState {
    static S: OnceLock<CapState> = OnceLock::new();
    S.get_or_init(|| CapState {
        status: Mutex::new(CapStatus::default()),
        child: tokio::sync::Mutex::new(None),
    })
}

/// Any compute process holding >1GB — training has priority for the card.
fn gpu_busy() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let heavy: Vec<String> = s
        .lines()
        .filter(|l| {
            l.rsplit(',')
                .next()
                .and_then(|m| m.trim().split(' ').next())
                .and_then(|n| n.parse::<u64>().ok())
                .map(|mib| mib > 1024)
                .unwrap_or(false)
        })
        .map(|l| l.trim().to_string())
        .collect();
    if heavy.is_empty() { None } else { Some(heavy.join("; ")) }
}

#[derive(Deserialize)]
pub struct RunReq {
    folder: String,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u64,
    #[serde(default)]
    skip_existing: bool,
    #[serde(default)]
    one_sentence: bool,
}
fn default_model() -> String { "Qwen/Qwen3-VL-4B-Instruct".into() }
fn default_max_tokens() -> u64 { 512 }

pub async fn run(Json(req): Json<RunReq>) -> (StatusCode, Json<Value>) {
    let st = state();

    if !req.folder.starts_with("/home/alex/") || req.folder.contains("..") {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "folder out of scope (must be under /home/alex, no ..)"})));
    }
    if !std::path::Path::new(&req.folder).is_dir() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": format!("not a folder: {}", req.folder)})));
    }
    if st.child.lock().await.is_some() {
        return (StatusCode::CONFLICT, Json(json!({"error": "a caption job is already running"})));
    }
    if let Some(who) = gpu_busy() {
        return (StatusCode::CONFLICT, Json(json!({"error": format!("GPU busy (training has priority): {who}")})));
    }

    let log = std::fs::File::create(STDERR_LOG).ok();
    let mut cmd = Command::new(PY);
    cmd.arg("-u")
        .arg(SCRIPT)
        .arg("--folder").arg(&req.folder)
        .arg("--model").arg(&req.model)
        .arg("--max-tokens").arg(req.max_tokens.to_string());
    if let Some(p) = req.prompt.as_ref().filter(|p| !p.trim().is_empty()) {
        cmd.arg("--prompt").arg(p);
    }
    if req.skip_existing { cmd.arg("--skip-existing"); }
    if req.one_sentence { cmd.arg("--one-sentence"); }
    cmd.stdout(Stdio::piped());
    match log.and_then(|f| f.try_clone().ok()) {
        Some(f) => { cmd.stderr(Stdio::from(f)); }
        None => { cmd.stderr(Stdio::null()); }
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("spawn captioner.py: {e}")}))),
    };
    let pid = child.id();
    let stdout = child.stdout.take().unwrap();
    {
        let mut s = st.status.lock().unwrap();
        *s = CapStatus {
            running: true,
            folder: req.folder.clone(),
            model: req.model.clone(),
            ..Default::default()
        };
    }
    *st.child.lock().await = Some(child);

    tokio::spawn(async move {
        let st = state();
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let Some(rest) = line.strip_prefix("CAPJSON ") else { continue };
            let Ok(ev) = serde_json::from_str::<Value>(rest) else { continue };
            let ty = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let mut s = st.status.lock().unwrap();
            match ty {
                "start" => {
                    s.total = ev.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                    s.found = ev.get("found").and_then(|v| v.as_u64()).unwrap_or(0);
                }
                "file_start" => {
                    s.current_file = ev.get("file").and_then(|v| v.as_str()).unwrap_or("").to_string();
                }
                "progress" => {
                    s.done = ev.get("done").and_then(|v| v.as_u64()).unwrap_or(s.done);
                    let file = ev.get("file").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let caption = ev.get("caption").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let sidecar = ev.get("sidecar").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    s.last_caption = caption.clone();
                    s.current_file = file.clone();
                    s.results.push(Result { file, caption, sidecar });
                    if s.results.len() > 200 {
                        let drop = s.results.len() - 200;
                        s.results.drain(0..drop);
                    }
                }
                "error" => {
                    let file = ev.get("file").and_then(|v| v.as_str()).unwrap_or("");
                    let err = ev.get("error").and_then(|v| v.as_str()).unwrap_or("");
                    s.error = format!("{file}: {err}");
                }
                "fatal" => {
                    s.fatal = ev.get("error").and_then(|v| v.as_str()).unwrap_or("").to_string();
                }
                "done" => {
                    s.done = ev.get("done").and_then(|v| v.as_u64()).unwrap_or(s.done);
                    s.finished = true;
                }
                _ => {}
            }
        }
        // stdout closed -> reap and clear running
        {
            let mut ch = st.child.lock().await;
            if let Some(c) = ch.as_mut() {
                let _ = c.wait().await;
            }
            *ch = None;
        }
        let mut s = st.status.lock().unwrap();
        s.running = false;
        if !s.finished && s.fatal.is_empty() && !s.aborted {
            s.fatal = "captioner exited before finishing (see captioner_last.log)".into();
        }
    });

    (StatusCode::OK, Json(json!({"started": true, "pid": pid, "folder": req.folder, "model": req.model})))
}

pub async fn status() -> Json<Value> {
    let s = state().status.lock().unwrap().clone();
    Json(json!({ "status": s }))
}

pub async fn abort() -> (StatusCode, Json<Value>) {
    let st = state();
    let mut ch = st.child.lock().await;
    if let Some(c) = ch.as_mut() {
        let _ = c.start_kill();
    }
    *ch = None;
    let mut s = st.status.lock().unwrap();
    s.running = false;
    s.aborted = true;
    (StatusCode::OK, Json(json!({"aborted": true})))
}
