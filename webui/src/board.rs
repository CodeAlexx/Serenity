// board.rs — SerenityBoard, built INTO the serenity web trainer supervisor.
// A single Rust binary = training UI + live metrics board, no Python in the path.
//
// Ported from the Python SerenityBoard (/home/alex/serenityboard). The original
// keeps one board.db PER run directory and a server that scans a logdir; here we
// consolidate into ONE db (webui/board.db) with a run-discriminated scalars table
// so the ported frontend's REST calls map 1:1 (see the schema-mapping table in the
// handoff). The frontend is served verbatim at /board with a fetch/WebSocket shim
// (board/board_boot.js) that namespaces /api/* -> /api/board/* and /ws/live.
//
// Two ingestion sources feed the same tables, deduped by primary key:
//   (a) LIVE hook  — the supervisor's stdout pump hands every parsed progress line
//                    to board::ingest (web-launched runs).
//   (b) TAILER     — a tokio task scanning output/*/train_{cli,web}.log for active
//                    files (mtime < 60s), re-parsing new lines through the SAME
//                    crate::parse_progress (covers CLI runs launched outside the UI).
// Run identity  = basename(workspace_dir); scalar identity = (run, tag, step).
// INSERT OR IGNORE makes both sources idempotent; a terminal status set by
// run_ended is never downgraded by the tailer.

use std::{
    collections::HashMap,
    path::{Path as FsPath, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use rusqlite::{params, Connection};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;

const OUTPUT_DIR: &str = "/home/alex/mojodiffusion/output";
// scalar tags emitted from a parsed progress line (grouped by "/" in the UI)
const TAG_LOSS: &str = "loss/train_step";
const TAG_GRAD: &str = "grad_norm/train_step";
const TAG_SPS: &str = "perf/s_per_step";
const TAG_EPOCH: &str = "progress/epoch";
// a "running" run whose newest point is older than this reads as "stopped"
const STALE_S: f64 = 120.0;
// a log file is a live training run if touched within this many seconds
const TAILER_ACTIVE_S: u64 = 60;
// skip the backlog of a large pre-existing log the first time we see it
const TAILER_BACKLOG_CAP: u64 = 2 * 1024 * 1024;

#[derive(Clone)]
struct LiveEvent {
    run: String,
    tag: String,
    json: String, // pre-serialized frontend message, sent verbatim over the socket
}

struct Board {
    db: Mutex<Connection>,
    board_dir: PathBuf,
    tx: broadcast::Sender<LiveEvent>,
}

static BOARD: OnceLock<Board> = OnceLock::new();

fn board() -> &'static Board {
    BOARD.get().expect("board::init() not called")
}

fn now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn basename(path: &str) -> String {
    FsPath::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

// ── initialization ─────────────────────────────────────────────────────────

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runs (
    name              TEXT PRIMARY KEY,
    workspace_dir     TEXT NOT NULL,
    source            TEXT NOT NULL,
    preset_id         TEXT,
    status            TEXT NOT NULL,
    start_time        REAL NOT NULL,
    last_wall_time    REAL,
    last_step         INTEGER,
    max_steps         INTEGER,
    active_session_id TEXT
);
CREATE TABLE IF NOT EXISTS scalars (
    run       TEXT    NOT NULL,
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    value     REAL    NOT NULL,
    PRIMARY KEY (run, tag, step)
);
CREATE TABLE IF NOT EXISTS run_notes (
    run        TEXT PRIMARY KEY,
    note       TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scalars_run_tag_step ON scalars(run, tag, step);
";

/// Open webui/board.db, create tables, and spawn the workspace-log tailer.
/// Called once from main() inside the tokio runtime.
pub fn init() {
    let repo = crate::REPO_ROOT;
    let db_path = format!("{repo}/webui/board.db");
    let board_dir = PathBuf::from(format!("{repo}/webui/board"));

    let conn = Connection::open(&db_path).expect("open board.db");
    conn.pragma_update(None, "journal_mode", "WAL").ok();
    conn.pragma_update(None, "synchronous", "NORMAL").ok();
    conn.pragma_update(None, "busy_timeout", 5000).ok();
    conn.execute_batch(SCHEMA).expect("board schema");

    let (tx, _rx) = broadcast::channel(1024);
    let _ = BOARD.set(Board {
        db: Mutex::new(conn),
        board_dir,
        tx,
    });

    tokio::spawn(tailer_loop());
    println!("serenity board on /board (db {db_path})");
}

// ── ingestion (shared by live hook + tailer) ────────────────────────────────

/// INSERT OR IGNORE one scalar point; broadcast a live event iff a new row landed.
fn insert_point(conn: &Connection, run: &str, tag: &str, step: u64, wt: f64, value: f64, sid: &str) {
    if !value.is_finite() {
        return;
    }
    let n = conn
        .execute(
            "INSERT OR IGNORE INTO scalars (run, tag, step, wall_time, value) VALUES (?, ?, ?, ?, ?)",
            params![run, tag, step as i64, wt, value],
        )
        .unwrap_or(0);
    if n > 0 {
        let msg = json!({
            "type": "scalar",
            "run": run,
            "tag": tag,
            "session_id": sid,
            "points": [{"step": step, "wall_time": wt, "value": value}],
        })
        .to_string();
        let _ = board().tx.send(LiveEvent {
            run: run.to_string(),
            tag: tag.to_string(),
            json: msg,
        });
    }
}

/// Core ingest: write the four scalar tags for one parsed step and refresh the run row.
/// `source` marks who observed it first; the run row is upserted, never resurrected
/// out of a terminal status.
fn ingest_step(
    conn: &Connection,
    run: &str,
    workspace: &str,
    source: &str,
    step: u64,
    epoch: u64,
    loss: f64,
    grad_norm: f64,
    s_per_step: f64,
    max_steps: u64,
) {
    let wt = now_s();
    let sid = session_id(conn, run, workspace, source, max_steps);
    if step > 0 {
        insert_point(conn, run, TAG_LOSS, step, wt, loss, &sid);
        insert_point(conn, run, TAG_GRAD, step, wt, grad_norm, &sid);
        if s_per_step > 0.0 {
            insert_point(conn, run, TAG_SPS, step, wt, s_per_step, &sid);
        }
        insert_point(conn, run, TAG_EPOCH, step, wt, epoch as f64, &sid);
    }
    // refresh liveness; bump status to running only if it was already running
    conn.execute(
        "UPDATE runs SET last_wall_time = ?, last_step = ?, max_steps = CASE WHEN ? > 0 THEN ? ELSE max_steps END WHERE name = ?",
        params![wt, step as i64, max_steps as i64, max_steps as i64, run],
    )
    .ok();
    conn.execute(
        "UPDATE runs SET status = 'running' WHERE name = ? AND status = 'running'",
        params![run],
    )
    .ok();
}

/// Ensure a run row exists and return its session id. Creates the row (status
/// running) on first sight; never overwrites an existing row here.
fn session_id(conn: &Connection, run: &str, workspace: &str, source: &str, max_steps: u64) -> String {
    let existing: Option<String> = conn
        .query_row(
            "SELECT active_session_id FROM runs WHERE name = ?",
            params![run],
            |r| r.get(0),
        )
        .ok()
        .flatten();
    if let Some(sid) = existing {
        return sid;
    }
    let start = now_s();
    let sid = format!("s{}", (start * 1000.0) as i64);
    conn.execute(
        "INSERT OR IGNORE INTO runs (name, workspace_dir, source, preset_id, status, start_time, max_steps, active_session_id) \
         VALUES (?, ?, ?, NULL, 'running', ?, ?, ?)",
        params![run, workspace, source, start, max_steps as i64, sid],
    )
    .ok();
    sid
}

/// LIVE hook — called from the stdout pump for every parsed progress line of a
/// web-launched run. `workspace` is the run's absolute workspace dir.
pub fn ingest(workspace: &str, step: u64, epoch: u64, loss: f64, grad_norm: f64, s_per_step: f64, max_steps: u64) {
    if BOARD.get().is_none() {
        return;
    }
    let run = basename(workspace);
    let conn = board().db.lock().unwrap();
    ingest_step(&conn, &run, workspace, "web", step, epoch, loss, grad_norm, s_per_step, max_steps);
}

/// Called at run launch — upsert the run row as running (web source).
pub fn run_started(workspace: &str, preset_id: &str, max_steps: u64) {
    if BOARD.get().is_none() {
        return;
    }
    let run = basename(workspace);
    let start = now_s();
    let sid = format!("s{}", (start * 1000.0) as i64);
    let conn = board().db.lock().unwrap();
    conn.execute(
        "INSERT INTO runs (name, workspace_dir, source, preset_id, status, start_time, max_steps, active_session_id) \
         VALUES (?, ?, 'web', ?, 'running', ?, ?, ?) \
         ON CONFLICT(name) DO UPDATE SET \
           workspace_dir = excluded.workspace_dir, source = 'web', preset_id = excluded.preset_id, \
           status = 'running', start_time = excluded.start_time, max_steps = excluded.max_steps, \
           active_session_id = excluded.active_session_id, last_wall_time = NULL, last_step = NULL",
        params![run, workspace, preset_id, start, max_steps as i64, sid],
    )
    .ok();
}

/// Called at run end (reap) — set the terminal status. "exited" reads as "completed".
pub fn run_ended(workspace: &str, status: &str) {
    if BOARD.get().is_none() {
        return;
    }
    let run = basename(workspace);
    let mapped = if status == "exited" { "completed" } else { status };
    let conn = board().db.lock().unwrap();
    conn.execute(
        "UPDATE runs SET status = ? WHERE name = ?",
        params![mapped, run],
    )
    .ok();
}

// ── workspace-log tailer ────────────────────────────────────────────────────

struct TailState {
    offset: u64,
    scratch: crate::RunInfo,
    workspace: String,
    source: &'static str,
}

async fn tailer_loop() {
    let mut tracked: HashMap<String, TailState> = HashMap::new();
    loop {
        scan_once(&mut tracked);
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

fn scan_once(tracked: &mut HashMap<String, TailState>) {
    let Ok(rd) = std::fs::read_dir(OUTPUT_DIR) else {
        return;
    };
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let workspace = dir.to_string_lossy().to_string();
        for (fname, source) in [("train_cli.log", "cli"), ("train_web.log", "web")] {
            let log = dir.join(fname);
            let Ok(meta) = std::fs::metadata(&log) else {
                continue;
            };
            // active = touched recently
            let fresh = meta
                .modified()
                .ok()
                .and_then(|m| SystemTime::now().duration_since(m).ok())
                .map(|d| d.as_secs() < TAILER_ACTIVE_S)
                .unwrap_or(false);
            if !fresh {
                continue;
            }
            let len = meta.len();
            let key = log.to_string_lossy().to_string();
            let st = tracked.entry(key.clone()).or_insert_with(|| TailState {
                // skip a large pre-existing backlog; otherwise read from the top
                offset: if len > TAILER_BACKLOG_CAP { len } else { 0 },
                scratch: fresh_runinfo(&workspace),
                workspace: workspace.clone(),
                source,
            });
            if len < st.offset {
                st.offset = 0; // file truncated / rotated
            }
            tail_file(&log, st);
        }
    }
}

fn tail_file(log: &FsPath, st: &mut TailState) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(log) else {
        return;
    };
    if f.seek(SeekFrom::Start(st.offset)).is_err() {
        return;
    }
    let mut buf = String::new();
    let Ok(n) = f.read_to_string(&mut buf) else {
        return;
    };
    if n == 0 {
        return;
    }
    // only consume up to the last newline; keep the partial tail for next scan
    let consumed = match buf.rfind('\n') {
        Some(i) => i + 1,
        None => return,
    };
    let run = basename(&st.workspace);
    let conn = board().db.lock().unwrap();
    for line in buf[..consumed].lines() {
        if crate::parse_progress(line, &mut st.scratch) {
            let s = &st.scratch;
            ingest_step(
                &conn, &run, &st.workspace, st.source,
                s.step, s.epoch, s.loss, s.grad_norm, s.s_per_step, s.total_steps,
            );
        }
    }
    st.offset += consumed as u64;
}

/// A blank RunInfo scratch to feed crate::parse_progress; only the progress
/// fields it touches matter.
fn fresh_runinfo(workspace: &str) -> crate::RunInfo {
    crate::RunInfo {
        id: 0,
        preset_id: String::new(),
        backend: String::new(),
        workspace_dir: workspace.to_string(),
        log_path: String::new(),
        status: "running".into(),
        pid: None,
        step: 0,
        total_steps: 0,
        epoch: 0,
        total_epochs: 0,
        loss: 0.0,
        grad_norm: 0.0,
        s_per_step: 0.0,
        eta: String::new(),
    }
}

// ── REST endpoints (ported from serenityboard data_provider/app/routes) ──────

fn scalar_rows(conn: &Connection, run: &str, tag: &str) -> Vec<(i64, f64, f64)> {
    let mut out = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT step, wall_time, value FROM scalars WHERE run = ? AND tag = ? ORDER BY step",
    ) {
        if let Ok(rows) = stmt.query_map(params![run, tag], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?))
        }) {
            for row in rows.flatten() {
                out.push(row);
            }
        }
    }
    out
}

/// Port of read_scalars_downsampled: n<=0 => full; else evenly sample n keeping
/// first + last.
fn downsample(rows: Vec<(i64, f64, f64)>, n: i64) -> Vec<(i64, f64, f64)> {
    if n <= 0 || (rows.len() as i64) <= n {
        return rows;
    }
    let n = n.max(3) as usize;
    let total = rows.len();
    let mut idx: Vec<usize> = Vec::with_capacity(n);
    idx.push(0);
    for i in 1..(n - 1) {
        let j = (i as f64) * (total as f64 - 1.0) / (n as f64 - 1.0);
        idx.push(j.round() as usize);
    }
    idx.push(total - 1);
    idx.dedup();
    idx.into_iter().map(|i| rows[i]).collect()
}

fn project_xaxis(rows: &[(i64, f64, f64)], x_axis: &str) -> Value {
    let t0 = rows.first().map(|r| r.1).unwrap_or(0.0);
    let arr: Vec<Value> = rows
        .iter()
        .map(|&(step, wt, v)| match x_axis {
            "wall_time" => json!([wt, wt, v]),
            "relative" => json!([wt - t0, wt, v]),
            _ => json!([step, wt, v]),
        })
        .collect();
    Value::Array(arr)
}

async fn list_runs() -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let now = now_s();
    let mut out: Vec<Value> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT name, source, status, start_time, last_wall_time, last_step, max_steps, active_session_id \
         FROM runs ORDER BY start_time DESC",
    ) {
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f64>(3)?,
                r.get::<_, Option<f64>>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, Option<i64>>(6)?,
                r.get::<_, Option<String>>(7)?,
            ))
        });
        if let Ok(rows) = rows {
            for row in rows.flatten() {
                let (name, source, mut status, start, last_wt, last_step, max_steps, sid) = row;
                if status == "running" {
                    if let Some(wt) = last_wt {
                        if now - wt > STALE_S {
                            status = "stopped".into();
                        }
                    }
                }
                let hparams = match max_steps {
                    Some(ms) => json!({ "max_steps": ms }),
                    None => json!({}),
                };
                out.push(json!({
                    "name": name,
                    "source": source,
                    "status": status,
                    "start_time": start,
                    "last_activity": last_wt,
                    "last_step": last_step,
                    "max_steps": max_steps,
                    "active_session_id": sid,
                    "hparams": hparams,
                }));
            }
        }
    }
    Json(Value::Array(out))
}

async fn tags(Path(run): Path<String>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let mut scalars: Vec<String> = Vec::new();
    if let Ok(mut stmt) =
        conn.prepare("SELECT DISTINCT tag FROM scalars WHERE run = ? ORDER BY tag")
    {
        if let Ok(rows) = stmt.query_map(params![run], |r| r.get::<_, String>(0)) {
            scalars = rows.flatten().collect();
        }
    }
    Json(json!({
        "scalars": scalars,
        "tensors": [], "artifacts": [], "text_events": [], "audio": [],
        "trace_events": [], "eval_suites": [], "pr_curves": [],
        "graphs": [], "meshes": [], "embeddings": [],
    }))
}

async fn metrics(Path(run): Path<String>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let mut scalars: Vec<Value> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tag, COUNT(*), MAX(step), MAX(wall_time) FROM scalars WHERE run = ? GROUP BY tag ORDER BY tag",
    ) {
        if let Ok(rows) = stmt.query_map(params![run], |r| {
            Ok(json!({
                "tag": r.get::<_, String>(0)?,
                "count": r.get::<_, i64>(1)?,
                "last_step": r.get::<_, Option<i64>>(2)?,
                "last_wall_time": r.get::<_, Option<f64>>(3)?,
            }))
        }) {
            scalars = rows.flatten().collect();
        }
    }
    Json(json!({
        "scalars": scalars,
        "tensors": [], "artifacts": [], "text_events": [], "audio": [],
        "pr_curves": [], "graphs": [], "meshes": [], "embeddings": [],
    }))
}

#[derive(Deserialize)]
struct ScalarQ {
    tag: String,
    #[serde(default)]
    downsample: Option<i64>,
    #[serde(default)]
    x_axis: Option<String>,
}

async fn scalars(Path(run): Path<String>, Query(q): Query<ScalarQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let rows = scalar_rows(&conn, &run, &q.tag);
    let rows = downsample(rows, q.downsample.unwrap_or(5000));
    Json(project_xaxis(&rows, q.x_axis.as_deref().unwrap_or("step")))
}

#[derive(Deserialize)]
struct TagsQ {
    tags: String,
}

async fn scalars_last(Path(run): Path<String>, Query(q): Query<TagsQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let mut out = serde_json::Map::new();
    for tag in q.tags.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let row = conn
            .query_row(
                "SELECT step, wall_time, value FROM scalars WHERE run = ? AND tag = ? ORDER BY step DESC LIMIT 1",
                params![run, tag],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?)),
            )
            .ok();
        if let Some((step, wt, v)) = row {
            out.insert(tag.to_string(), json!({"step": step, "wall_time": wt, "value": v}));
        }
    }
    Json(Value::Object(out))
}

#[derive(Deserialize)]
struct CompareQ {
    tag: String,
    runs: String,
    #[serde(default)]
    downsample: Option<i64>,
    #[serde(default)]
    x_axis: Option<String>,
}

async fn compare_scalars(Query(q): Query<CompareQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let x_axis = q.x_axis.as_deref().unwrap_or("step");
    let mut out = serde_json::Map::new();
    for run in q.runs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let rows = scalar_rows(&conn, run, &q.tag);
        let rows = downsample(rows, q.downsample.unwrap_or(5000));
        out.insert(run.to_string(), project_xaxis(&rows, x_axis));
    }
    Json(Value::Object(out))
}

async fn get_notes(Path(run): Path<String>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let row = conn
        .query_row(
            "SELECT note, updated_at FROM run_notes WHERE run = ?",
            params![run],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
        )
        .ok();
    match row {
        Some((note, updated)) => Json(json!({"note": note, "updated_at": updated})),
        None => Json(json!({"note": "", "updated_at": Value::Null})),
    }
}

#[derive(Deserialize)]
struct NoteBody {
    note: String,
}

async fn put_notes(Path(run): Path<String>, Json(b): Json<NoteBody>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    conn.execute(
        "INSERT INTO run_notes (run, note, updated_at) VALUES (?, ?, ?) \
         ON CONFLICT(run) DO UPDATE SET note = excluded.note, updated_at = excluded.updated_at",
        params![run, b.note, now_s()],
    )
    .ok();
    Json(json!({"ok": true}))
}

async fn delete_run(Path(run): Path<String>) -> Json<Value> {
    // Board-only delete: drop the metrics rows. We do NOT touch the training
    // workspace on disk (that's the run's real output, out of board scope).
    let conn = board().db.lock().unwrap();
    conn.execute("DELETE FROM scalars WHERE run = ?", params![run]).ok();
    conn.execute("DELETE FROM run_notes WHERE run = ?", params![run]).ok();
    let n = conn.execute("DELETE FROM runs WHERE name = ?", params![run]).unwrap_or(0);
    if n > 0 {
        Json(json!({"deleted": run}))
    } else {
        Json(json!({"error": "run not found"}))
    }
}

async fn empty_arr() -> Json<Value> {
    Json(json!([]))
}
async fn empty_obj() -> Json<Value> {
    Json(json!({}))
}
async fn null_json() -> Json<Value> {
    Json(Value::Null)
}

// ── live WebSocket ───────────────────────────────────────────────────────────

async fn ws_live(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_ws)
}

async fn handle_ws(mut socket: WebSocket) {
    let mut rx = board().tx.subscribe();
    let mut sub_runs: Vec<String> = Vec::new();
    let mut sub_tags: Vec<String> = vec!["*".into()];
    loop {
        tokio::select! {
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Text(t))) => {
                        if let Ok(v) = serde_json::from_str::<Value>(&t) {
                            if let Some(sub) = v.get("subscribe") {
                                sub_runs = str_vec(sub.get("runs"));
                                let tags = str_vec(sub.get("tags"));
                                sub_tags = if tags.is_empty() { vec!["*".into()] } else { tags };
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
            ev = rx.recv() => {
                match ev {
                    Ok(ev) => {
                        let run_ok = sub_runs.iter().any(|r| r == &ev.run);
                        let tag_ok = sub_tags.iter().any(|p| glob_match(p, &ev.tag));
                        if run_ok && tag_ok {
                            if socket.send(Message::Text(ev.json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}

fn str_vec(v: Option<&Value>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn glob_match(pat: &str, s: &str) -> bool {
    if pat == "*" {
        return true;
    }
    if !pat.contains('*') {
        return pat == s;
    }
    let parts: Vec<&str> = pat.split('*').collect();
    let mut pos = 0usize;
    let last = parts.len() - 1;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !s[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if i == last {
            if !s[pos..].ends_with(part) {
                return false;
            }
        } else {
            match s[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

// ── static frontend at /board ────────────────────────────────────────────────

async fn board_index() -> Response {
    serve_file(&board().board_dir.join("index.html"))
}

async fn board_asset(Path(file): Path<String>) -> Response {
    if file.contains("..") {
        return StatusCode::FORBIDDEN.into_response();
    }
    serve_file(&board().board_dir.join(&file))
}

fn serve_file(path: &FsPath) -> Response {
    match std::fs::read(path) {
        Ok(bytes) => {
            let ct = content_type(path);
            ([(header::CONTENT_TYPE, ct)], bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

fn content_type(path: &FsPath) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

// ── router (merged into the supervisor's app) ────────────────────────────────

pub fn router() -> Router<std::sync::Arc<crate::AppState>> {
    Router::new()
        .route("/board", get(board_index))
        .route("/board/", get(board_index))
        .route("/board/*file", get(board_asset))
        .route("/api/board/runs", get(list_runs))
        .route("/api/board/runs/:run", axum::routing::delete(delete_run))
        .route("/api/board/runs/:run/tags", get(tags))
        .route("/api/board/runs/:run/metrics", get(metrics))
        .route("/api/board/runs/:run/scalars", get(scalars))
        .route("/api/board/runs/:run/scalars/last", get(scalars_last))
        .route("/api/board/runs/:run/notes", get(get_notes).put(put_notes))
        .route("/api/board/runs/:run/traces", get(empty_arr))
        .route("/api/board/runs/:run/eval", get(empty_arr))
        .route("/api/board/runs/:run/artifacts", get(empty_arr))
        .route("/api/board/runs/:run/pr-curves", get(empty_arr))
        .route("/api/board/runs/:run/audio", get(empty_arr))
        .route("/api/board/runs/:run/histograms", get(empty_arr))
        .route("/api/board/runs/:run/distributions", get(empty_arr))
        .route("/api/board/runs/:run/embeddings", get(empty_arr))
        .route("/api/board/runs/:run/custom-scalars/layout", get(null_json))
        .route("/api/board/runs/:run/custom-scalars/data", get(empty_obj))
        .route("/api/board/compare/scalars", get(compare_scalars))
        .route("/api/board/compare/hparams", get(empty_obj))
        .route("/api/board/ws/live", get(ws_live))
}
