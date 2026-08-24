//! Reference toolkit tools (cwd-jailed filesystem and shell).
//!
//! [`default_toolkit`] installs a **sandboxed** shell
//! ([`ShellTool::sandboxed`] with [`ovo_sandbox::PrefixExecPolicy::workspace_shell`]).
//! Use [`trusted_toolkit`] with [`ovo_sandbox::TrustedExecution`] only when
//! opting out of process isolation.

#![forbid(unsafe_code)]

pub mod exec_session;
pub mod glob_files;
pub mod grep;
pub mod jail;
pub mod path_util;
pub mod read_file;
pub mod shell;
pub mod write_file;

use std::path::PathBuf;
use std::sync::Arc;

pub use exec_session::ExecSessionTool;
pub use glob_files::{GlobTool, glob_match};
pub use grep::GrepTool;
use ovo_sandbox::{
    PrefixExecPolicy, SandboxBackend, SandboxError, SandboxPolicy, TrustedExecution,
};
use ovo_tools::SharedTool;
pub use path_util::{PathJailError, resolve_jailed};
pub use read_file::ReadFileTool;
pub use shell::ShellTool;
pub use write_file::WriteFileTool;

/// Convenience bundle: read / write / grep / glob / **sandboxed** shell.
///
/// `jail` is canonicalized. Shell uses [`ShellTool::sandboxed`] with
/// [`ovo_sandbox::PrefixExecPolicy::workspace_shell`]. This bundle does not
/// include an exec-session tool.
///
/// # Errors
///
/// Returns [`ovo_sandbox::SandboxError::Failed`] when `jail` cannot be
/// canonicalized.
pub fn default_toolkit(
    jail: impl Into<PathBuf>,
    backend: Arc<dyn SandboxBackend>,
) -> Result<Vec<SharedTool>, SandboxError> {
    let root =
        std::fs::canonicalize(jail.into()).map_err(|e| SandboxError::Failed(e.to_string()))?;
    let policy = SandboxPolicy::workspace(root.clone());
    Ok(vec![
        Arc::new(ReadFileTool::with_jail(root.clone())),
        Arc::new(WriteFileTool::with_jail(root.clone())),
        Arc::new(GrepTool::with_jail(root.clone())),
        Arc::new(GlobTool::with_jail(root.clone())),
        Arc::new(
            ShellTool::sandboxed(root, backend, policy)
                .with_exec_policy(Arc::new(PrefixExecPolicy::workspace_shell())),
        ),
    ])
}

/// Same five tools as [`default_toolkit`], with a **trusted** (unsandboxed) shell.
#[must_use]
pub fn trusted_toolkit(jail: impl Into<PathBuf>, _t: TrustedExecution) -> Vec<SharedTool> {
    let root = jail.into();
    vec![
        Arc::new(ReadFileTool::with_jail(root.clone())),
        Arc::new(WriteFileTool::with_jail(root.clone())),
        Arc::new(GrepTool::with_jail(root.clone())),
        Arc::new(GlobTool::with_jail(root.clone())),
        Arc::new(ShellTool::trusted(root, TrustedExecution)),
    ]
}

/// Installed OS sandbox backend for this target and feature set.
///
/// No `NoSandbox` fallback. `full` without `seatbelt`/`landlock` fails.
///
/// # Errors
///
/// Seatbelt construction does not fail; wrap may still fail at spawn.
#[cfg(all(feature = "seatbelt", target_os = "macos"))]
pub fn platform_sandbox() -> Result<Arc<dyn SandboxBackend>, SandboxError> {
    Ok(Arc::new(ovo_sandbox::SeatbeltBackend))
}

/// Installed OS sandbox backend for this target and feature set.
///
/// # Errors
///
/// Returns [`ovo_sandbox::SandboxError::Failed`] when the `ovo-landlock` helper
/// cannot be resolved.
#[cfg(all(feature = "landlock", target_os = "linux"))]
pub fn platform_sandbox() -> Result<Arc<dyn SandboxBackend>, SandboxError> {
    Ok(Arc::new(ovo_sandbox::LandlockBackend::new()?))
}

/// Installed OS sandbox backend for this target and feature set.
///
/// # Errors
///
/// Always [`ovo_sandbox::SandboxError::Failed`]: no OS backend is compiled in.
#[cfg(not(any(
    all(feature = "seatbelt", target_os = "macos"),
    all(feature = "landlock", target_os = "linux")
)))]
pub fn platform_sandbox() -> Result<Arc<dyn SandboxBackend>, SandboxError> {
    Err(SandboxError::Failed(
        "no OS sandbox backend for this target/feature set".into(),
    ))
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "unit tests use expect for setup"
)]
mod tests {
    use std::sync::Arc;

    use ovo_sandbox::NoSandbox;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn default_toolkit_relative_path_after_create_dir_all() {
        let tmp = tempfile::Builder::new()
            .prefix("ovo-toolkit-jail-")
            .tempdir_in(".")
            .expect("tmp");
        let cwd = std::env::current_dir().expect("cwd");
        let rel = tmp
            .path()
            .strip_prefix(&cwd)
            .expect("tmp under cwd")
            .to_path_buf();
        assert!(rel.is_relative(), "{rel:?}");
        let tools = default_toolkit(&rel, Arc::new(NoSandbox)).expect("relative jail");
        assert_eq!(tools.len(), 5);
        assert!(
            !tools.iter().any(|t| t.name() == "exec_session"),
            "default_toolkit must not include exec_session"
        );
    }

    #[test]
    fn default_toolkit_missing_path_err() {
        let err = default_toolkit(
            "ovo-toolkit-missing-jail-do-not-create",
            Arc::new(NoSandbox),
        );
        assert!(
            matches!(err, Err(SandboxError::Failed(_))),
            "missing jail must fail"
        );
    }

    #[test]
    fn default_toolkit_has_no_exec_session() {
        let dir = tempdir().expect("tmp");
        let tools = default_toolkit(dir.path(), Arc::new(NoSandbox)).expect("jail");
        assert!(!tools.iter().any(|t| t.name() == "exec_session"));
        let names: Vec<_> = tools.iter().map(|t| t.name().to_owned()).collect();
        assert_eq!(names, ["read_file", "write_file", "grep", "glob", "shell"]);
    }
}
