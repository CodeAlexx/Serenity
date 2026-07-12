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
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

const PY: &str = "/home/alex/ai-toolkit/venv/bin/python";
const SCRIPT: &str = "/home/alex/serenity-trainer/webui/captioner.py";
const STDERR_LOG: &str = "/home/alex/serenity-trainer/webui/captioner_last.log";
// engine="mojo": pure-Mojo one-shot captioner (Qwen3-VL). Contract:
//   qwen3vl_caption <image> [prompt] [max_new]
// prints the caption between a `=== CAPTION ===` line and a `=== END ===` line.
// The binary is single-image, so the loop over the folder lives here in Rust.
const MOJO_CAP: &str = "/home/alex/mojodiffusion/output/bin/qwen3vl_caption";
// Prompt resolution mirrors captioner.py so both engines produce the same
// training-style caption when the user leaves the Prompt field blank.
const DEFAULT_PROMPT: &str = "Write a detailed caption for this image to train an image-generation model. Describe the main subject, appearance, pose, clothing, expression, any action, the setting and background, lighting, colors, composition, and the art medium or style. Output only the caption itself with no preamble.";
const ONE_SENTENCE_PROMPT: &str = "Describe this image in one detailed sentence for an image-generation training caption. Output only the sentence, no preamble.";

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
                // match launch()'s guard: unparseable nvidia-smi memory = treat as busy (refuse)
                .unwrap_or(true)
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
    /// "python" (ai-toolkit venv, default) | "mojo" (pure-Mojo qwen3vl_caption)
    #[serde(default = "default_engine")]
    engine: String,
}
fn default_model() -> String { "Qwen/Qwen3-VL-4B-Instruct".into() }
fn default_max_tokens() -> u64 { 512 }
fn default_engine() -> String { "python".into() }

pub async fn run(Json(req): Json<RunReq>) -> (StatusCode, Json<Value>) {
    let st = state();

    if !req.folder.starts_with("/home/alex/") || req.folder.contains("..") {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "folder out of scope (must be under /home/alex, no ..)"})));
    }
    if !std::path::Path::new(&req.folder).is_dir() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": format!("not a folder: {}", req.folder)})));
    }
    // `child` covers the python path (one long-lived child) AND the mojo path's
    // current per-image child; `running` closes the gap BETWEEN mojo images where
    // `child` is briefly None but a job is still active.
    if st.child.lock().await.is_some() || st.status.lock().unwrap().running {
        return (StatusCode::CONFLICT, Json(json!({"error": "a caption job is already running"})));
    }
    if let Some(who) = gpu_busy() {
        return (StatusCode::CONFLICT, Json(json!({"error": format!("GPU busy (training has priority): {who}")})));
    }

    // ── engine="mojo": drive the single-image qwen3vl_caption binary in a loop ──
    if req.engine == "mojo" {
        return run_mojo(req).await;
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

/// Pull the caption out of qwen3vl_caption stdout: the lines strictly between the
/// `=== CAPTION ===` and `=== END ===` markers, joined with '\n' and trimmed. The
/// binary also prints bucket/timing chatter around the block — this drops it.
fn extract_caption(stdout: &str) -> String {
    let mut in_cap = false;
    let mut cap: Vec<&str> = vec![];
    for line in stdout.lines() {
        let t = line.trim();
        if t == "=== CAPTION ===" {
            in_cap = true;
            continue;
        }
        if t == "=== END ===" {
            break;
        }
        if in_cap {
            cap.push(line);
        }
    }
    cap.join("\n").trim().to_string()
}

/// engine="mojo" path: enumerate images in the folder and drive the single-image
/// qwen3vl_caption binary once per image, writing .txt sidecars and updating the
/// SAME CapStatus the python path drives. Abort kills the current child; the loop
/// then stops at the next image boundary. Caller has already run every shared
/// guard (folder scope, is_dir, single-job mutex, gpu_busy).
async fn run_mojo(req: RunReq) -> (StatusCode, Json<Value>) {
    let st = state();
    if !std::path::Path::new(MOJO_CAP).exists() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "mojo captioner binary not built"})));
    }

    // images directly under the folder (png/jpg/jpeg/webp, case-insensitive), sorted
    let mut imgs: Vec<PathBuf> = vec![];
    if let Ok(rd) = std::fs::read_dir(&req.folder) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_file() {
                continue;
            }
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
            if ["png", "jpg", "jpeg", "webp"].contains(&ext.as_str()) {
                imgs.push(p);
            }
        }
    }
    imgs.sort();
    let found = imgs.len() as u64;
    let mut todo: Vec<PathBuf> = vec![];
    for img in imgs {
        let txt = img.with_extension("txt");
        if req.skip_existing && txt.exists() && std::fs::metadata(&txt).map(|m| m.len() > 0).unwrap_or(false) {
            continue;
        }
        todo.push(img);
    }
    let total = todo.len() as u64;

    // prompt resolution mirrors captioner.py exactly (blank field -> training default)
    let prompt = if let Some(p) = req.prompt.as_ref().filter(|p| !p.trim().is_empty()) {
        let mut p = p.trim().to_string();
        if req.one_sentence {
            p.push_str(" Respond in a single sentence.");
        }
        p
    } else if req.one_sentence {
        ONE_SENTENCE_PROMPT.to_string()
    } else {
        DEFAULT_PROMPT.to_string()
    };
    let max_new = req.max_tokens;

    {
        let mut s = st.status.lock().unwrap();
        *s = CapStatus {
            running: true,
            folder: req.folder.clone(),
            model: "qwen3vl_caption (mojo)".into(),
            total,
            found,
            ..Default::default()
        };
    }

    tokio::spawn(async move {
        let st = state();
        let mut done: u64 = 0;
        for img in todo {
            if st.status.lock().unwrap().aborted {
                break;
            }
            let base = img.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            {
                let mut s = st.status.lock().unwrap();
                s.current_file = base.clone();
            }
            let mut cmd = Command::new(MOJO_CAP);
            cmd.arg(&img)
                .arg(&prompt)
                .arg(max_new.to_string())
                .env("LD_LIBRARY_PATH", crate::CUDA_LD)
                .stdout(Stdio::piped())
                .stderr(Stdio::null());
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    st.status.lock().unwrap().error = format!("{base}: spawn: {e}");
                    continue;
                }
            };
            let Some(stdout) = child.stdout.take() else {
                st.status.lock().unwrap().error = format!("{base}: no stdout");
                continue;
            };
            *st.child.lock().await = Some(child);
            // collect stdout, then pull the caption from between the two markers
            let mut lines = BufReader::new(stdout).lines();
            let mut raw = String::new();
            while let Ok(Some(line)) = lines.next_line().await {
                raw.push_str(&line);
                raw.push('\n');
            }
            // reap the per-image child
            {
                let mut ch = st.child.lock().await;
                if let Some(c) = ch.as_mut() {
                    let _ = c.wait().await;
                }
                *ch = None;
            }
            if st.status.lock().unwrap().aborted {
                break;
            }
            let cap = extract_caption(&raw);
            if cap.is_empty() {
                st.status.lock().unwrap().error = format!("{base}: no caption produced");
                continue;
            }
            let txt = img.with_extension("txt");
            match std::fs::write(&txt, &cap) {
                Ok(_) => {
                    done += 1;
                    let sidecar = txt.to_string_lossy().to_string();
                    let mut s = st.status.lock().unwrap();
                    s.done = done;
                    s.last_caption = cap.clone();
                    s.results.push(Result { file: base.clone(), caption: cap, sidecar });
                    if s.results.len() > 200 {
                        let drop = s.results.len() - 200;
                        s.results.drain(0..drop);
                    }
                }
                Err(e) => {
                    st.status.lock().unwrap().error = format!("{base}: write failed: {e}");
                }
            }
        }
        // clear any lingering child handle + finalize
        *st.child.lock().await = None;
        let mut s = st.status.lock().unwrap();
        s.running = false;
        if !s.aborted {
            s.finished = true;
        }
    });

    (StatusCode::OK, Json(json!({"started": true, "engine": "mojo", "folder": req.folder, "total": total, "found": found})))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_caption_strips_chatter_around_markers() {
        // shape verified from qwen3vl_caption.mojo:151-260 — bucket line before,
        // timing line after; caption is the single print between the markers.
        let stdout = "[caption] bucket grid: 32 x 32  (S = 1024 )\n\
                      [caption] prefix tokens: 12  image_pad: 256  nvis: 256\n\
                      === CAPTION ===\n\
                      A red fox sitting in tall grass at golden hour.\n\
                      === END ===\n\
                      timing: preprocess 0.1 s | vision tower 0.4 s\n";
        assert_eq!(
            extract_caption(stdout),
            "A red fox sitting in tall grass at golden hour."
        );
    }

    #[test]
    fn extract_caption_multiline_joined_and_trimmed() {
        let stdout = "=== CAPTION ===\nline one\nline two\n=== END ===\n";
        assert_eq!(extract_caption(stdout), "line one\nline two");
    }

    #[test]
    fn extract_caption_empty_when_no_markers() {
        // a crash before generation prints no CAPTION block -> empty (per-file err)
        assert_eq!(extract_caption("Unhandled exception: CUDA OOM\n"), "");
        assert_eq!(extract_caption(""), "");
    }
}
