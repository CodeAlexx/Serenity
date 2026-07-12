// Cache-prep endpoints for the serenity web trainer.
//
// The missing link between "raw folder + captions" and "launch a run". A prepare
// recipe (presets.json `prepare.steps`) is a STEP LIST, not one command: each step
// is its own process — a python `interp` + script, or a Mojo `bin` — with its own
// cwd/env, run SEQUENTIALLY. Each step must exit 0 AND write its `produces` file
// before the next runs. For krea2 that is two steps:
//   A. raw folder -> staged   (CPU, ai-toolkit python: krea2_stage_images.py)
//   B. staged     -> cache     (GPU, Mojo: krea2_prepare_cache)
// Structure mirrors captioner.rs — self-contained OnceLock state, a single
// prepare-job mutex, and the SAME gpu_busy() >1GB refusal so training / caption /
// prepare never contend for the 24GB card (step B is GPU-heavy). Whole-job
// success = the FINAL step's produces (== the requested {out}) exists non-empty,
// at which point status.cache = {out} closes the raw-folder -> launch loop.

use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

// Read fresh on every run so an edited presets.json needs no server restart.
const PRESETS: &str = "/home/alex/serenity-trainer/webui/presets.json";

#[derive(Default, Clone, Serialize)]
struct PrepStatus {
    running: bool,
    finished: bool,
    aborted: bool,
    preset: String,
    stage_dir: String,
    out: String,
    done: u64,       // completed steps
    total: u64,      // total steps in the chain
    current: String, // "step k/N: <last stdout line>"
    error: String,   // reserved for non-fatal notes
    fatal: String,   // step failure / missing final cache
    cache: String,   // produced cache path (set on success — closes the loop)
}

struct PrepState {
    status: Mutex<PrepStatus>,
    child: tokio::sync::Mutex<Option<Child>>,
}

fn state() -> &'static PrepState {
    static S: OnceLock<PrepState> = OnceLock::new();
    S.get_or_init(|| PrepState {
        status: Mutex::new(PrepStatus::default()),
        child: tokio::sync::Mutex::new(None),
    })
}

/// Any compute process holding >1GB — training has priority for the card.
/// (Byte-identical to captioner::gpu_busy so all three jobs share one rule.)
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
                // unparseable nvidia-smi memory = treat as busy (refuse)
                .unwrap_or(true)
        })
        .map(|l| l.trim().to_string())
        .collect();
    if heavy.is_empty() { None } else { Some(heavy.join("; ")) }
}

#[derive(Deserialize)]
pub struct RunReq {
    preset_id: String,
    stage_dir: String, // the RAW image folder (N.jpg + N.txt) the user picks
    out: String,       // the FINAL cache .safetensors
    #[serde(default)]
    n: u64,
    #[serde(default = "default_size")]
    size: u64,
}
fn default_size() -> u64 { 512 }

/// One resolved step of the chain (placeholders already substituted).
#[derive(Debug)]
struct Step {
    program: String, // the interp OR the bin to exec
    is_bin: bool,    // wording for the "not built"/"not found" preflight error
    argv: Vec<String>,
    cwd: Option<String>,
    env: Vec<(String, String)>,
    produces: String, // the file this step MUST create (non-empty) to advance
}

/// Substitute the placeholders in one template slot. `prepare.steps` is DATA — a
/// step's argv/env/produces are interpolated only through these five keys.
fn subst(t: &str, stage_dir: &str, staged: &str, out: &str, n: u64, size: u64) -> String {
    t.replace("{stage_dir}", stage_dir)
        .replace("{staged}", staged)
        .replace("{out}", out)
        .replace("{n}", &n.to_string())
        .replace("{size}", &size.to_string())
}

/// Look up the `prepare` block for a preset from presets.json. Returns the block
/// on success, or a (status, message) the caller turns into an honest error:
///   404 unknown preset · 422 no prepare recipe · 500 presets.json unreadable.
fn prepare_block(preset_id: &str) -> Result<Value, (StatusCode, String)> {
    let raw = std::fs::read_to_string(PRESETS)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("read presets.json: {e}")))?;
    let v: Value = serde_json::from_str(&raw)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("parse presets.json: {e}")))?;
    let presets = v
        .get("presets")
        .and_then(|p| p.as_array())
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "presets.json has no presets array".to_string()))?;
    let p = presets
        .iter()
        .find(|p| p.get("id").and_then(|x| x.as_str()) == Some(preset_id))
        .ok_or((StatusCode::NOT_FOUND, format!("unknown preset {preset_id}")))?;
    match p.get("prepare") {
        Some(b) if b.is_object() => Ok(b.clone()),
        _ => Err((StatusCode::UNPROCESSABLE_ENTITY, format!("no prepare recipe for {preset_id}"))),
    }
}

/// Resolve the `prepare.steps` list into concrete Steps with all placeholders
/// substituted. Errors are 422s naming the offending step.
fn build_steps(
    block: &Value,
    stage_dir: &str,
    staged: &str,
    out: &str,
    n: u64,
    size: u64,
) -> Result<Vec<Step>, (StatusCode, String)> {
    let arr = block
        .get("steps")
        .and_then(|s| s.as_array())
        .ok_or((StatusCode::UNPROCESSABLE_ENTITY, "prepare recipe has no steps".to_string()))?;
    if arr.is_empty() {
        return Err((StatusCode::UNPROCESSABLE_ENTITY, "prepare recipe has empty steps".to_string()));
    }
    let mut steps = Vec::with_capacity(arr.len());
    for (i, sv) in arr.iter().enumerate() {
        let k = i + 1;
        let interp = sv.get("interp").and_then(|x| x.as_str());
        let bin = sv.get("bin").and_then(|x| x.as_str());
        let (program, is_bin) = match (interp, bin) {
            (Some(p), _) => (p.to_string(), false),
            (None, Some(b)) => (b.to_string(), true),
            _ => return Err((StatusCode::UNPROCESSABLE_ENTITY, format!("prepare step {k} has neither interp nor bin"))),
        };
        let argv_tmpl = sv
            .get("argv")
            .and_then(|a| a.as_array())
            .ok_or((StatusCode::UNPROCESSABLE_ENTITY, format!("prepare step {k} has no argv")))?;
        let argv: Vec<String> = argv_tmpl
            .iter()
            .map(|t| subst(t.as_str().unwrap_or(""), stage_dir, staged, out, n, size))
            .collect();
        let cwd = sv.get("cwd").and_then(|c| c.as_str()).map(|s| s.to_string());
        let mut env: Vec<(String, String)> = vec![];
        if let Some(e) = sv.get("env").and_then(|e| e.as_object()) {
            for (kk, vv) in e {
                env.push((kk.clone(), subst(vv.as_str().unwrap_or(""), stage_dir, staged, out, n, size)));
            }
        }
        let produces = subst(
            sv.get("produces").and_then(|p| p.as_str()).unwrap_or(""),
            stage_dir,
            staged,
            out,
            n,
            size,
        );
        if produces.is_empty() {
            return Err((StatusCode::UNPROCESSABLE_ENTITY, format!("prepare step {k} has no produces")));
        }
        steps.push(Step { program, is_bin, argv, cwd, env, produces });
    }
    Ok(steps)
}

pub async fn run(Json(req): Json<RunReq>) -> (StatusCode, Json<Value>) {
    let st = state();

    // scope validation (mirror captioner: under /home/alex, no traversal)
    if !req.stage_dir.starts_with("/home/alex/") || req.stage_dir.contains("..") {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "stage_dir out of scope (must be under /home/alex, no ..)"})));
    }
    if !Path::new(&req.stage_dir).is_dir() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": format!("not a folder: {}", req.stage_dir)})));
    }
    if !req.out.starts_with("/home/alex/") || req.out.contains("..") {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "out out of scope (must be under /home/alex, no ..)"})));
    }
    if req.n == 0 {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "n (image count) must be > 0"})));
    }

    // resolve recipe + derive the intermediate staged dir ( <dir of out>/_staged )
    let block = match prepare_block(&req.preset_id) {
        Ok(b) => b,
        Err((code, msg)) => return (code, Json(json!({"error": msg}))),
    };
    let out_parent = Path::new(&req.out)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new("/").to_path_buf());
    let staged = out_parent.join("_staged").to_string_lossy().to_string();
    let steps = match build_steps(&block, &req.stage_dir, &staged, &req.out, req.n, req.size) {
        Ok(s) => s,
        Err((code, msg)) => return (code, Json(json!({"error": msg}))),
    };
    // preflight: every step's program must exist (fail fast, before spawning any)
    for s in &steps {
        if !Path::new(&s.program).exists() {
            let msg = if s.is_bin {
                format!("prepare binary not built: {}", s.program)
            } else {
                format!("prepare interpreter not found: {}", s.program)
            };
            return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": msg})));
        }
    }

    // single-job + shared GPU guards (copied from captioner; `running` covers the
    // gaps BETWEEN steps where `child` is briefly None)
    if st.child.lock().await.is_some() || st.status.lock().unwrap().running {
        return (StatusCode::CONFLICT, Json(json!({"error": "a prepare job is already running"})));
    }
    if let Some(who) = gpu_busy() {
        return (StatusCode::CONFLICT, Json(json!({"error": format!("GPU busy (training has priority): {who}")})));
    }

    // dirs + combined chain log at <out>.prepare.log
    let _ = std::fs::create_dir_all(&staged);
    let _ = std::fs::create_dir_all(&out_parent);
    let log_path = format!("{}.prepare.log", req.out);
    let _ = std::fs::File::create(&log_path); // truncate/create

    let n_steps = steps.len() as u64;
    {
        let mut s = st.status.lock().unwrap();
        *s = PrepStatus {
            running: true,
            preset: req.preset_id.clone(),
            stage_dir: req.stage_dir.clone(),
            out: req.out.clone(),
            total: n_steps,
            ..Default::default()
        };
    }

    let log_path_task = log_path.clone();
    tokio::spawn(async move {
        let st = state();
        let mut ok = true;
        for (i, step) in steps.iter().enumerate() {
            if st.status.lock().unwrap().aborted {
                ok = false;
                break;
            }
            let k = i + 1;
            {
                let mut s = st.status.lock().unwrap();
                s.done = i as u64;
                s.current = format!("step {k}/{n_steps}: starting {}", step.program);
            }
            let stderr_f = std::fs::OpenOptions::new().append(true).open(&log_path_task).ok();
            let mut cmd = Command::new(&step.program);
            cmd.args(&step.argv).stdout(Stdio::piped());
            if let Some(cwd) = &step.cwd {
                cmd.current_dir(cwd);
            }
            for (kk, vv) in &step.env {
                cmd.env(kk, vv);
            }
            match stderr_f.and_then(|f| f.try_clone().ok()) {
                Some(f) => { cmd.stderr(Stdio::from(f)); }
                None => { cmd.stderr(Stdio::null()); }
            }
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    st.status.lock().unwrap().fatal = format!("step {k}/{n_steps}: spawn {} failed: {e}", step.program);
                    ok = false;
                    break;
                }
            };
            let stdout = child.stdout.take().unwrap();
            *st.child.lock().await = Some(child);

            let mut logw = std::fs::OpenOptions::new().append(true).open(&log_path_task).ok();
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(f) = logw.as_mut() {
                    let _ = writeln!(f, "{line}");
                }
                st.status.lock().unwrap().current = format!("step {k}/{n_steps}: {line}");
            }
            // reap (keep exit code for the failure message)
            let code = {
                let mut ch = st.child.lock().await;
                let code = if let Some(c) = ch.as_mut() {
                    c.wait().await.ok().and_then(|s| s.code())
                } else {
                    None
                };
                *ch = None;
                code
            };
            if st.status.lock().unwrap().aborted {
                ok = false;
                break;
            }
            // a step advances ONLY on exit 0 AND its produces file existing non-empty
            let produced = std::fs::metadata(&step.produces).map(|m| m.len() > 0).unwrap_or(false);
            if code != Some(0) || !produced {
                let mut s = st.status.lock().unwrap();
                s.fatal = format!(
                    "step {k}/{n_steps} ({}) exited {}{}",
                    step.program,
                    code.map(|c| c.to_string()).unwrap_or_else(|| "abnormally".into()),
                    if produced { String::new() } else { format!("; did not produce {}", step.produces) },
                );
                ok = false;
                break;
            }
            st.status.lock().unwrap().done = k as u64;
        }
        // clear any lingering child handle + finalize
        *st.child.lock().await = None;
        let mut s = st.status.lock().unwrap();
        s.running = false;
        if ok && !s.aborted {
            // whole-job success = the FINAL step's produces (== out) exists non-empty
            let produced = std::fs::metadata(&s.out).map(|m| m.len() > 0).unwrap_or(false);
            if produced {
                s.done = n_steps;
                s.finished = true;
                s.cache = s.out.clone();
            } else if s.fatal.is_empty() {
                s.fatal = format!("prepare finished but final cache {} is missing/empty", s.out);
            }
        }
    });

    (StatusCode::OK, Json(json!({
        "started": true,
        "preset": req.preset_id,
        "out": req.out,
        "staged": staged,
        "steps": n_steps,
        "log": log_path
    })))
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
    fn subst_fills_all_placeholders() {
        assert_eq!(subst("{stage_dir}", "/raw", "/st", "/o", 42, 512), "/raw");
        assert_eq!(subst("{staged}", "/raw", "/st", "/o", 42, 512), "/st");
        assert_eq!(subst("{out}", "/raw", "/st", "/o", 42, 512), "/o");
        assert_eq!(subst("{n}", "/raw", "/st", "/o", 42, 512), "42");
        assert_eq!(subst("{size}", "/raw", "/st", "/o", 42, 1024), "1024");
        assert_eq!(subst("literal", "/raw", "/st", "/o", 1, 512), "literal");
        // composed path (produces of step A)
        assert_eq!(subst("{staged}/images.safetensors", "/raw", "/st", "/o", 1, 512), "/st/images.safetensors");
    }

    #[test]
    fn krea2_steps_substitute_the_two_step_chain() {
        // exercises the REAL presets.json krea2 recipe end-to-end (resolve + build).
        let block = prepare_block("krea2").expect("krea2 prepare block");
        let steps = build_steps(
            &block,
            "/home/alex/raw",
            "/home/alex/out/_staged",
            "/home/alex/out/cache.safetensors",
            137,
            512,
        )
        .expect("krea2 steps build");
        assert_eq!(steps.len(), 2, "krea2 prepare is a two-step chain");

        // step A: ai-toolkit python staging, raw -> staged/images.safetensors
        assert!(steps[0].program.ends_with("/python"), "step A runs the venv python");
        assert!(!steps[0].is_bin);
        assert_eq!(
            steps[0].argv,
            vec![
                "serenitymojo/training/krea2_stage_images.py".to_string(),
                "/home/alex/raw".to_string(),
                "/home/alex/out/_staged".to_string(),
                "512".to_string(),
            ]
        );
        assert_eq!(steps[0].produces, "/home/alex/out/_staged/images.safetensors");
        assert_eq!(steps[0].cwd.as_deref(), Some("/home/alex/mojodiffusion"));

        // step B: Mojo cache bin, staged -> out, with LD_LIBRARY_PATH
        assert_eq!(steps[1].program, "/home/alex/mojodiffusion/output/bin/krea2_prepare_cache");
        assert!(steps[1].is_bin);
        assert_eq!(
            steps[1].argv,
            vec![
                "/home/alex/out/_staged".to_string(),
                "/home/alex/out/cache.safetensors".to_string(),
                "137".to_string(),
                "512".to_string(),
            ]
        );
        assert_eq!(steps[1].produces, "/home/alex/out/cache.safetensors");
        assert!(
            steps[1].env.iter().any(|(k, v)| k == "LD_LIBRARY_PATH" && v.contains("cshim/lib")),
            "step B carries the CUDA loader path"
        );
    }

    #[test]
    fn prepare_block_unknown_preset_is_404() {
        let (code, _) = prepare_block("does-not-exist").unwrap_err();
        assert_eq!(code, StatusCode::NOT_FOUND);
    }

    #[test]
    fn build_steps_errors_when_no_steps() {
        let block = json!({"default_size": 512}); // no `steps`
        let err = build_steps(&block, "/raw", "/st", "/o", 1, 512).unwrap_err();
        assert_eq!(err.0, StatusCode::UNPROCESSABLE_ENTITY);
    }
}
