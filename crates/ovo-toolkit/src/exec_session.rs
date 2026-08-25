//! Argv-only sandboxed process sessions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use ovo_sandbox::{DenyAllExecPolicy, ExecPolicy, SandboxBackend, SandboxPolicy};
use ovo_tools::error::codes;
use ovo_tools::stream::ToolStream;
use ovo_tools::{
    DynTool, ToolCallContext, ToolError, ToolMetadata, ToolProgress, ToolResult, with_progress,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Default per-`read` wait.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Default captured bytes per pipe (`stdout` / `stderr` independently).
pub const DEFAULT_MAX_OUTPUT: usize = 64 * 1024;
/// Default cap on live sessions owned by one tool instance.
pub const DEFAULT_MAX_SESSIONS: usize = 4;

/// Argv-only sandboxed process session tool.
///
/// Not included in [`crate::default_toolkit`]. There is no `Default` impl and
/// no trusted constructor: isolation is always a [`SandboxBackend`].
///
/// `command` is exec'd as argv (no `sh -c` wrapping). [`ExecPolicy`] is
/// consulted before wrap; [`DenyAllExecPolicy`] is the [`Self::sandboxed`]
/// default. Dropping the tool drops live children without awaiting (spawn
/// uses `kill_on_drop(true)`).
pub struct ExecSessionTool {
    jail_root: PathBuf,
    backend: Arc<dyn SandboxBackend>,
    policy: SandboxPolicy,
    exec_policy: Arc<dyn ExecPolicy>,
    timeout: Duration,
    max_output: usize,
    max_sessions: usize,
    sessions: Mutex<HashMap<String, Session>>,
}

struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    stderr: ChildStderr,
    cancel: CancellationToken,
}

/// Session taken out of the map for I/O. Reinserts on drop so aborting the
/// call (dispatch timeout / deadline) does not `kill_on_drop` the child.
struct HeldSession<'a> {
    tool: &'a ExecSessionTool,
    session_id: String,
    session: Option<Session>,
}

impl HeldSession<'_> {
    fn as_mut(&mut self) -> Result<&mut Session, ToolError> {
        self.session
            .as_mut()
            .ok_or_else(|| codes::execution("exec_session: session already consumed"))
    }

    fn discard(mut self) {
        self.session.take();
    }
}

impl Drop for HeldSession<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let _ = self.tool.put_session(self.session_id.clone(), session);
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExecAction {
    Start,
    Write,
    Read,
    Close,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecSessionArgs {
    action: ExecAction,
    command: Option<Vec<String>>,
    session_id: Option<String>,
    input: Option<String>,
}

impl ExecSessionTool {
    /// Sandboxed sessions: wrap via `backend` under `policy`.
    ///
    /// Defaults to [`DenyAllExecPolicy`]; use [`Self::with_exec_policy`] to allow argv.
    #[must_use]
    pub fn sandboxed(
        root: impl Into<PathBuf>,
        backend: Arc<dyn SandboxBackend>,
        policy: SandboxPolicy,
    ) -> Self {
        Self {
            jail_root: root.into(),
            backend,
            policy,
            exec_policy: Arc::new(DenyAllExecPolicy),
            timeout: DEFAULT_TIMEOUT,
            max_output: DEFAULT_MAX_OUTPUT,
            max_sessions: DEFAULT_MAX_SESSIONS,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Replace the argv allowlist consulted before wrap.
    #[must_use]
    pub fn with_exec_policy(mut self, policy: Arc<dyn ExecPolicy>) -> Self {
        self.exec_policy = policy;
        self
    }

    /// Set per-`read` wait.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set captured-byte cap per pipe.
    #[must_use]
    pub fn with_max_output(mut self, n: usize) -> Self {
        self.max_output = n;
        self
    }

    /// Set live-session cap.
    #[must_use]
    pub fn with_max_sessions(mut self, n: usize) -> Self {
        self.max_sessions = n;
        self
    }

    fn lock_sessions(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<String, Session>>, ToolError> {
        self.sessions
            .lock()
            .map_err(|_| codes::execution("exec_session: session lock poisoned"))
    }

    fn take_session(&self, session_id: &str) -> Result<Session, ToolError> {
        self.lock_sessions()?
            .remove(session_id)
            .ok_or_else(|| codes::invalid_args(format!("unknown session_id: {session_id}")))
    }

    fn put_session(&self, session_id: String, session: Session) -> Result<(), ToolError> {
        self.lock_sessions()?.insert(session_id, session);
        Ok(())
    }

    async fn dispatch(
        &self,
        ctx: ToolCallContext,
        args: ExecSessionArgs,
    ) -> Result<ToolResult, ToolError> {
        match args.action {
            ExecAction::Start => self.action_start(ctx, args),
            ExecAction::Write => self.action_write(ctx, args).await,
            ExecAction::Read => self.action_read(ctx, args).await,
            ExecAction::Close => self.action_close(args),
        }
    }

    fn action_start(
        &self,
        ctx: ToolCallContext,
        args: ExecSessionArgs,
    ) -> Result<ToolResult, ToolError> {
        if ctx.cancel.is_cancelled() {
            return Err(codes::cancelled());
        }
        let command = args
            .command
            .filter(|c| !c.is_empty())
            .ok_or_else(|| codes::invalid_args("start requires non-empty command"))?;
        match self.exec_policy.decide(&command) {
            ovo_sandbox::ExecDecision::Allow => {}
            ovo_sandbox::ExecDecision::Deny => {
                return Err(codes::denied("exec policy: deny"));
            }
        }
        {
            let g = self.lock_sessions()?;
            if g.len() >= self.max_sessions {
                return Err(codes::concurrency_limit("exec_session: max sessions"));
            }
        }
        let Some((program, rest)) = command.split_first() else {
            return Err(codes::invalid_args("start requires non-empty command"));
        };
        let mut cmd = Command::new(program);
        cmd.args(rest).current_dir(&self.jail_root);
        let mut cmd = self
            .backend
            .wrap(&self.policy, cmd)
            .map_err(|e| codes::execution(format!("sandbox '{}': {e}", self.backend.name())))?;
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| codes::execution(format!("spawn exec_session: {e}")))?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err(codes::execution("exec_session: stdio not piped"));
        };
        let session_id = Uuid::new_v4().to_string();
        let session = Session {
            child,
            stdin,
            stdout,
            stderr,
            cancel: ctx.cancel.child_token(),
        };
        {
            let mut g = self.lock_sessions()?;
            if g.len() >= self.max_sessions {
                drop(g);
                drop(session);
                return Err(codes::concurrency_limit("exec_session: max sessions"));
            }
            g.insert(session_id.clone(), session);
        }
        Ok(ToolResult {
            content: format!("session_id={session_id}"),
            structured: Some(json!({ "session_id": session_id })),
            is_error: false,
        })
    }

    async fn action_write(
        &self,
        ctx: ToolCallContext,
        args: ExecSessionArgs,
    ) -> Result<ToolResult, ToolError> {
        let session_id = require_session_id(args.session_id)?;
        let input = args
            .input
            .ok_or_else(|| codes::invalid_args("write requires input"))?;
        let session = self.take_session(&session_id)?;
        let mut held = HeldSession {
            tool: self,
            session_id,
            session: Some(session),
        };
        if ctx.cancel.is_cancelled() || held.as_mut()?.cancel.is_cancelled() {
            held.discard();
            return Err(codes::cancelled());
        }
        let write_res = {
            let session = held.as_mut()?;
            tokio::select! {
                () = ctx.cancel.cancelled() => None,
                () = session.cancel.cancelled() => None,
                r = session.stdin.write_all(input.as_bytes()) => Some(r),
            }
        };
        match write_res {
            None => {
                held.discard();
                Err(codes::cancelled())
            }
            Some(Ok(())) => Ok(ToolResult {
                content: "ok".to_owned(),
                structured: Some(json!({ "ok": true })),
                is_error: false,
            }),
            Some(Err(e)) => Err(codes::execution(format!("exec_session write: {e}"))),
        }
    }

    async fn action_read(
        &self,
        ctx: ToolCallContext,
        args: ExecSessionArgs,
    ) -> Result<ToolResult, ToolError> {
        let session_id = require_session_id(args.session_id)?;
        let session = self.take_session(&session_id)?;
        let mut held = HeldSession {
            tool: self,
            session_id,
            session: Some(session),
        };
        if ctx.cancel.is_cancelled() || held.as_mut()?.cancel.is_cancelled() {
            held.discard();
            return Err(codes::cancelled());
        }
        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();
        let outcome = {
            let session = held.as_mut()?;
            tokio::time::timeout(
                self.timeout,
                capture_pipes(
                    session,
                    &mut stdout_buf,
                    &mut stderr_buf,
                    self.max_output,
                    &ctx.cancel,
                ),
            )
            .await
        };
        match outcome {
            Ok(Ok(CaptureEnd::Exited(status))) => {
                held.discard();
                Ok(read_result(stdout_buf, stderr_buf, true, !status.success()))
            }
            Ok(Ok(CaptureEnd::Cancelled)) => {
                held.discard();
                Err(codes::cancelled())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(read_result(stdout_buf, stderr_buf, false, false)),
        }
    }

    fn action_close(&self, args: ExecSessionArgs) -> Result<ToolResult, ToolError> {
        let session_id = require_session_id(args.session_id)?;
        drop(self.take_session(&session_id)?);
        Ok(ToolResult {
            content: "closed".to_owned(),
            structured: Some(json!({ "closed": true })),
            is_error: false,
        })
    }
}

impl Drop for ExecSessionTool {
    fn drop(&mut self) {
        if let Ok(mut g) = self.sessions.lock() {
            g.clear();
        }
    }
}

impl std::fmt::Debug for ExecSessionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let session_count = self.sessions.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("ExecSessionTool")
            .field("jail_root", &self.jail_root)
            .field("backend", &self.backend.name())
            .field("timeout", &self.timeout)
            .field("max_sessions", &self.max_sessions)
            .field("session_count", &session_count)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl DynTool for ExecSessionTool {
    fn name(&self) -> &'static str {
        "exec_session"
    }

    fn description(&self) -> &'static str {
        "Manage a sandboxed argv process session. Actions: start, write, read, close."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["start", "write", "read", "close"] },
                "command": {
                    "type": "array",
                    "items": { "type": "string" },
                    "minItems": 1,
                    "description": "argv for action=start; not a shell string"
                },
                "session_id": { "type": "string", "description": "uuid v4 from start" },
                "input": { "type": "string", "description": "bytes to stdin for action=write" }
            },
            "required": ["action"],
            "additionalProperties": false
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::shell_execute(self.timeout)
    }

    async fn call(&self, ctx: ToolCallContext, arguments: Value) -> Result<ToolResult, ToolError> {
        let stream = self.execute(ctx, arguments).await;
        ovo_tools::drain_terminal(stream).await
    }

    async fn execute(&self, ctx: ToolCallContext, arguments: Value) -> ToolStream {
        let args = match serde_json::from_value::<ExecSessionArgs>(arguments) {
            Ok(a) => a,
            Err(e) => {
                return ovo_tools::terminal_only(Err(codes::invalid_args(e.to_string())));
            }
        };
        let label = match args.action {
            ExecAction::Start => "start",
            ExecAction::Write => "write",
            ExecAction::Read => "read",
            ExecAction::Close => "close",
        };
        let progress = vec![ToolProgress::text(format!("exec_session: {label}"))];
        let result = self.dispatch(ctx, args).await;
        with_progress(progress, move || async move { result })
    }
}

fn require_session_id(session_id: Option<String>) -> Result<String, ToolError> {
    session_id
        .filter(|s| !s.is_empty())
        .ok_or_else(|| codes::invalid_args("exec_session requires session_id"))
}

fn append_capped(buf: &mut Vec<u8>, chunk: &[u8], max: usize) {
    if buf.len() >= max {
        return;
    }
    let take = max.saturating_sub(buf.len()).min(chunk.len());
    if let Some(part) = chunk.get(..take) {
        buf.extend_from_slice(part);
    }
}

fn read_result(stdout: Vec<u8>, stderr: Vec<u8>, exited: bool, is_error: bool) -> ToolResult {
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr).into_owned();
    let mut content = format!("exited={exited}\n");
    if !stdout.is_empty() {
        content.push_str("--- stdout ---\n");
        content.push_str(&stdout);
        if !content.ends_with('\n') {
            content.push('\n');
        }
    }
    if !stderr.is_empty() {
        content.push_str("--- stderr ---\n");
        content.push_str(&stderr);
    }
    ToolResult {
        content,
        structured: Some(json!({
            "stdout": stdout,
            "stderr": stderr,
            "exited": exited,
        })),
        is_error,
    }
}

enum CaptureEnd {
    Exited(ExitStatus),
    Cancelled,
}

enum PipeEvent {
    Cancel,
    Exit(std::io::Result<ExitStatus>),
    Stdout(std::io::Result<usize>),
    Stderr(std::io::Result<usize>),
}

async fn capture_pipes(
    session: &mut Session,
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
    max_output: usize,
    call_cancel: &CancellationToken,
) -> Result<CaptureEnd, ToolError> {
    let mut tmp_out = [0u8; 4096];
    let mut tmp_err = [0u8; 4096];
    let mut stdout_eof = false;
    let mut stderr_eof = false;
    let mut exit_status = None;
    loop {
        if let Some(status) = exit_status
            && stdout_eof
            && stderr_eof
        {
            return Ok(CaptureEnd::Exited(status));
        }
        let event = tokio::select! {
            () = call_cancel.cancelled() => PipeEvent::Cancel,
            () = session.cancel.cancelled() => PipeEvent::Cancel,
            r = session.child.wait(), if exit_status.is_none() => PipeEvent::Exit(r),
            r = session.stdout.read(&mut tmp_out), if !stdout_eof => PipeEvent::Stdout(r),
            r = session.stderr.read(&mut tmp_err), if !stderr_eof => PipeEvent::Stderr(r),
        };
        match event {
            PipeEvent::Cancel => return Ok(CaptureEnd::Cancelled),
            PipeEvent::Exit(r) => {
                exit_status =
                    Some(r.map_err(|e| codes::execution(format!("exec_session wait: {e}")))?);
            }
            PipeEvent::Stdout(r) => {
                apply_read(r, stdout_buf, &tmp_out, max_output, &mut stdout_eof)?;
            }
            PipeEvent::Stderr(r) => {
                apply_read(r, stderr_buf, &tmp_err, max_output, &mut stderr_eof)?;
            }
        }
    }
}

fn apply_read(
    result: std::io::Result<usize>,
    buf: &mut Vec<u8>,
    tmp: &[u8],
    max_output: usize,
    eof: &mut bool,
) -> Result<(), ToolError> {
    match result {
        Ok(0) => {
            *eof = true;
            Ok(())
        }
        Ok(n) => {
            if let Some(chunk) = tmp.get(..n) {
                append_capped(buf, chunk, max_output);
            }
            Ok(())
        }
        Err(e) => Err(codes::execution(format!("exec_session pipe: {e}"))),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use ovo_sandbox::{AllowAllExecPolicy, NoSandbox, SandboxError};
    use ovo_tools::{DispatchRequest, ToolDispatch, ToolRegistry};
    use ovo_types::{Deadline, ErrorCode, ToolCall, ToolCallId};
    use tempfile::tempdir;

    use super::*;

    struct CountingBackend {
        wraps: AtomicUsize,
    }

    impl SandboxBackend for CountingBackend {
        fn name(&self) -> &'static str {
            "counting"
        }

        fn wrap(&self, _policy: &SandboxPolicy, cmd: Command) -> Result<Command, SandboxError> {
            self.wraps.fetch_add(1, Ordering::SeqCst);
            let std = cmd.as_std();
            let mut copied = Command::new(std.get_program());
            copied.args(std.get_args());
            if let Some(dir) = std.get_current_dir() {
                copied.current_dir(dir);
            }
            for (k, v) in std.get_envs() {
                if let Some(val) = v {
                    copied.env(k, val);
                }
            }
            copied.kill_on_drop(true);
            Ok(copied)
        }
    }

    fn session_tool(root: &std::path::Path) -> ExecSessionTool {
        ExecSessionTool::sandboxed(root, Arc::new(NoSandbox), SandboxPolicy::workspace(root))
            .with_exec_policy(Arc::new(AllowAllExecPolicy))
            .with_timeout(Duration::from_secs(2))
    }

    fn ctx(root: &std::path::Path) -> ToolCallContext {
        ToolCallContext {
            cwd: Some(root.to_path_buf()),
            ..ToolCallContext::default()
        }
    }

    fn structured(r: &ToolResult) -> &Value {
        r.structured.as_ref().expect("structured")
    }

    fn session_id(r: &ToolResult) -> String {
        structured(r)
            .get("session_id")
            .and_then(Value::as_str)
            .expect("session_id")
            .to_owned()
    }

    async fn start_cmd(
        tool: &ExecSessionTool,
        call_ctx: ToolCallContext,
        command: &[&str],
    ) -> ToolResult {
        let command: Vec<Value> = command.iter().map(|s| Value::from(*s)).collect();
        tool.call(call_ctx, json!({ "action": "start", "command": command }))
            .await
            .expect("start")
    }

    #[tokio::test]
    async fn start_echo_read_exits() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path());
        let call_ctx = ctx(dir.path());
        let started = start_cmd(&tool, call_ctx.clone(), &["echo", "hi"]).await;
        let id = session_id(&started);
        let read = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let r = tool
                    .call(
                        call_ctx.clone(),
                        json!({ "action": "read", "session_id": id }),
                    )
                    .await
                    .expect("read");
                let st = structured(&r);
                let stdout = st.get("stdout").and_then(Value::as_str).unwrap_or("");
                let exited = st.get("exited").and_then(Value::as_bool).unwrap_or(false);
                if stdout.contains("hi") && exited {
                    return r;
                }
                assert!(!exited, "exited without hi: {}", r.content);
            }
        })
        .await
        .expect("echo read");
        assert!(
            structured(&read)
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("hi"))
        );
        let err = tool
            .call(call_ctx, json!({ "action": "read", "session_id": id }))
            .await
            .expect_err("gone");
        assert_eq!(err.code(), ErrorCode::ToolInvalidArgs);
    }

    #[tokio::test]
    async fn write_read_split_pipes() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path());
        let call_ctx = ctx(dir.path());
        let started = start_cmd(
            &tool,
            call_ctx.clone(),
            &["/bin/sh", "-c", "printf out; printf err >&2"],
        )
        .await;
        let id = session_id(&started);
        let read = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let r = tool
                    .call(
                        call_ctx.clone(),
                        json!({ "action": "read", "session_id": id }),
                    )
                    .await
                    .expect("read");
                let st = structured(&r);
                let stdout = st.get("stdout").and_then(Value::as_str).unwrap_or("");
                let stderr = st.get("stderr").and_then(Value::as_str).unwrap_or("");
                if stdout.contains("out") && stderr.contains("err") {
                    return r;
                }
                let exited = st.get("exited").and_then(Value::as_bool).unwrap_or(false);
                assert!(
                    !exited || (stdout.contains("out") && stderr.contains("err")),
                    "split failed: {}",
                    r.content
                );
                if exited {
                    return r;
                }
            }
        })
        .await
        .expect("split read");
        let st = structured(&read);
        assert!(
            st.get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("out")),
            "{}",
            read.content
        );
        assert!(
            st.get("stderr")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("err")),
            "{}",
            read.content
        );
    }

    #[tokio::test]
    async fn close_then_write_invalid() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path());
        let call_ctx = ctx(dir.path());
        let started = start_cmd(&tool, call_ctx.clone(), &["/bin/sleep", "30"]).await;
        let id = session_id(&started);
        tool.call(
            call_ctx.clone(),
            json!({ "action": "close", "session_id": id }),
        )
        .await
        .expect("close");
        let err = tool
            .call(
                call_ctx,
                json!({ "action": "write", "session_id": id, "input": "x" }),
            )
            .await
            .expect_err("closed");
        assert_eq!(err.code(), ErrorCode::ToolInvalidArgs);
    }

    #[tokio::test]
    async fn unknown_id_is_invalid_args() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path());
        let call_ctx = ctx(dir.path());
        for arguments in [
            json!({ "action": "read", "session_id": "not-a-uuid" }),
            json!({ "action": "write", "session_id": "not-a-uuid", "input": "x" }),
            json!({ "action": "close", "session_id": "not-a-uuid" }),
        ] {
            let err = tool
                .call(call_ctx.clone(), arguments)
                .await
                .expect_err("unknown");
            assert_eq!(err.code(), ErrorCode::ToolInvalidArgs);
        }
    }

    #[tokio::test]
    async fn max_sessions_concurrency() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path()).with_max_sessions(1);
        let call_ctx = ctx(dir.path());
        let started = start_cmd(&tool, call_ctx.clone(), &["/bin/sleep", "30"]).await;
        let id = session_id(&started);
        let err = tool
            .call(
                call_ctx.clone(),
                json!({ "action": "start", "command": ["echo", "x"] }),
            )
            .await
            .expect_err("limit");
        assert_eq!(err.code(), ErrorCode::ToolConcurrencyLimit);
        tool.call(
            call_ctx.clone(),
            json!({ "action": "close", "session_id": id }),
        )
        .await
        .expect("close");
        start_cmd(&tool, call_ctx, &["echo", "ok"]).await;
    }

    #[tokio::test]
    async fn start_cancel_kills_child() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path());
        let call_ctx = ctx(dir.path());
        tokio::time::timeout(Duration::from_secs(2), async {
            let started = start_cmd(&tool, call_ctx.clone(), &["/bin/sleep", "30"]).await;
            let id = session_id(&started);
            call_ctx.cancel.cancel();
            let err = tool
                .call(
                    call_ctx.clone(),
                    json!({ "action": "read", "session_id": id }),
                )
                .await
                .expect_err("cancelled");
            assert_eq!(err.code(), ErrorCode::ToolCancelled);
            for arguments in [
                json!({ "action": "read", "session_id": id }),
                json!({ "action": "write", "session_id": id, "input": "x" }),
                json!({ "action": "close", "session_id": id }),
            ] {
                let follow = tool
                    .call(call_ctx.clone(), arguments)
                    .await
                    .expect_err("gone");
                assert!(
                    matches!(
                        follow.code(),
                        ErrorCode::ToolInvalidArgs | ErrorCode::ToolCancelled
                    ),
                    "{follow:?}"
                );
            }
        })
        .await
        .expect("cancel must not hang");
    }

    #[tokio::test]
    async fn write_cancel_kills_child() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path());
        let call_ctx = ctx(dir.path());
        tokio::time::timeout(Duration::from_secs(2), async {
            let started = start_cmd(&tool, call_ctx.clone(), &["/bin/sleep", "30"]).await;
            let id = session_id(&started);
            let input = "x".repeat(256 * 1024);
            let write = tool.call(
                call_ctx.clone(),
                json!({ "action": "write", "session_id": id, "input": input }),
            );
            tokio::pin!(write);
            let still_pending = tokio::time::timeout(Duration::from_millis(50), &mut write)
                .await
                .is_err();
            assert!(still_pending, "write completed before cancel");
            call_ctx.cancel.cancel();
            let err = write.await.expect_err("cancelled");
            assert_eq!(err.code(), ErrorCode::ToolCancelled);
            let follow = tool
                .call(call_ctx, json!({ "action": "read", "session_id": id }))
                .await
                .expect_err("gone");
            assert!(
                matches!(
                    follow.code(),
                    ErrorCode::ToolInvalidArgs | ErrorCode::ToolCancelled
                ),
                "{follow:?}"
            );
        })
        .await
        .expect("write cancel must not hang");
    }

    #[tokio::test]
    async fn read_timeout_keeps_session() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path()).with_timeout(Duration::from_millis(200));
        let call_ctx = ctx(dir.path());
        let started = start_cmd(&tool, call_ctx.clone(), &["/bin/sleep", "30"]).await;
        let id = session_id(&started);
        let read = tool
            .call(
                call_ctx.clone(),
                json!({ "action": "read", "session_id": id }),
            )
            .await
            .expect("read");
        assert_eq!(
            structured(&read).get("exited").and_then(Value::as_bool),
            Some(false),
            "{}",
            read.content
        );
        tool.call(call_ctx, json!({ "action": "close", "session_id": id }))
            .await
            .expect("session still live");
    }

    #[tokio::test]
    async fn dispatch_deadline_restores_session() {
        let dir = tempdir().expect("temp");
        let tool = Arc::new(session_tool(dir.path()).with_timeout(Duration::from_millis(200)));
        let call_ctx = ctx(dir.path());
        let started = start_cmd(tool.as_ref(), call_ctx.clone(), &["/bin/sleep", "30"]).await;
        let id = session_id(&started);
        let shared: Arc<dyn DynTool> = tool.clone();
        let registry = ToolRegistry::from_tools(vec![shared]);
        let read_ctx = ToolCallContext {
            cwd: Some(dir.path().to_path_buf()),
            deadline: Some(Deadline::after(Duration::from_millis(50))),
            ..ToolCallContext::default()
        };
        let outs = tokio::time::timeout(
            Duration::from_secs(2),
            ToolDispatch::default().execute_batch(
                &registry,
                read_ctx,
                vec![DispatchRequest {
                    call: ToolCall {
                        id: ToolCallId::new("r1").expect("id"),
                        name: "exec_session".into(),
                        arguments: json!({ "action": "read", "session_id": id }),
                    },
                }],
            ),
        )
        .await
        .expect("dispatch must not hang");
        let err = outs
            .first()
            .expect("one")
            .result
            .as_ref()
            .expect_err("timeout");
        assert_eq!(err.code(), ErrorCode::ToolTimeout);
        let follow = tool
            .call(
                call_ctx.clone(),
                json!({ "action": "read", "session_id": id }),
            )
            .await
            .expect("session survived dispatch abort");
        assert_eq!(
            structured(&follow).get("exited").and_then(Value::as_bool),
            Some(false),
            "{}",
            follow.content
        );
        tool.call(call_ctx, json!({ "action": "close", "session_id": id }))
            .await
            .expect("close live session");
    }

    #[tokio::test]
    async fn deny_does_not_wrap() {
        let dir = tempdir().expect("temp");
        let backend = Arc::new(CountingBackend {
            wraps: AtomicUsize::new(0),
        });
        let isolation: Arc<dyn SandboxBackend> = backend.clone();
        let tool =
            ExecSessionTool::sandboxed(dir.path(), isolation, SandboxPolicy::workspace(dir.path()));
        let err = tool
            .call(
                ctx(dir.path()),
                json!({ "action": "start", "command": ["echo", "hi"] }),
            )
            .await
            .expect_err("denied");
        assert_eq!(err.code(), ErrorCode::ToolDenied);
        assert_eq!(backend.wraps.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    #[cfg(any(
        all(feature = "seatbelt", target_os = "macos"),
        all(feature = "landlock", target_os = "linux"),
    ))]
    async fn cfg_os_outside_cat() {
        let dir = tempdir().expect("temp");
        let root = dir.path().canonicalize().expect("canon");
        let outside = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME")
            .join(format!("ovo_execsess_out_{}", std::process::id()));
        std::fs::write(&outside, b"secret").expect("out");
        let outside_s = outside.to_string_lossy().into_owned();

        #[cfg(all(feature = "seatbelt", target_os = "macos"))]
        let tool = {
            use ovo_sandbox::SeatbeltBackend;
            ExecSessionTool::sandboxed(
                root.clone(),
                Arc::new(SeatbeltBackend),
                SandboxPolicy::workspace(root.clone()),
            )
            .with_exec_policy(Arc::new(AllowAllExecPolicy))
            .with_timeout(Duration::from_secs(2))
        };
        #[cfg(all(feature = "landlock", target_os = "linux"))]
        let tool = {
            use ovo_sandbox::{FsPolicy, LandlockBackend, NetPolicy};
            let Some(helper) = std::env::var_os("OVO_LANDLOCK_HELPER") else {
                let _ = std::fs::remove_file(&outside);
                return;
            };
            ExecSessionTool::sandboxed(
                root.clone(),
                Arc::new(LandlockBackend::with_helper(helper)),
                SandboxPolicy {
                    fs: FsPolicy::ReadWrite {
                        paths: vec![root.clone()],
                    },
                    net: NetPolicy::Allowed,
                },
            )
            .with_exec_policy(Arc::new(AllowAllExecPolicy))
            .with_timeout(Duration::from_secs(2))
        };

        let call_ctx = ctx(&root);
        let started = start_cmd(&tool, call_ctx.clone(), &["cat", &outside_s]).await;
        let id = session_id(&started);
        let read = tool
            .call(call_ctx, json!({ "action": "read", "session_id": id }))
            .await
            .expect("read");
        let _ = std::fs::remove_file(&outside);
        let stdout = structured(&read)
            .get("stdout")
            .and_then(Value::as_str)
            .unwrap_or("");
        assert!(
            !stdout.contains("secret"),
            "outside read leaked: {}",
            read.content
        );
        let exited = structured(&read)
            .get("exited")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        assert!(
            read.is_error || exited,
            "outside cat must fail: {}",
            read.content
        );
    }

    #[tokio::test]
    async fn read_both_pipes_no_deadlock() {
        let dir = tempdir().expect("temp");
        let tool = session_tool(dir.path()).with_max_output(16);
        let call_ctx = ctx(dir.path());
        let read = tokio::time::timeout(Duration::from_secs(2), async {
            let started = start_cmd(
                &tool,
                call_ctx.clone(),
                &[
                    "/bin/sh",
                    "-c",
                    "dd if=/dev/zero bs=65536 count=2; dd if=/dev/zero bs=65536 count=2 >&2",
                ],
            )
            .await;
            let id = session_id(&started);
            tool.call(call_ctx, json!({ "action": "read", "session_id": id }))
                .await
                .expect("read")
        })
        .await
        .expect("dual-pipe read must not deadlock");
        let st = structured(&read);
        let stdout = st.get("stdout").and_then(Value::as_str).unwrap_or("");
        let stderr = st.get("stderr").and_then(Value::as_str).unwrap_or("");
        assert_eq!(stdout.len(), 16, "stdout truncated: {}", read.content);
        assert_eq!(stderr.len(), 16, "stderr truncated: {}", read.content);
    }
}
