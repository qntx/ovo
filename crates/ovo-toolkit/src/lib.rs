//! Reference toolkit tools (cwd-jailed filesystem and shell).
//!
//! Shell requires an explicit isolation choice ([`ShellTool::trusted`] or
//! [`ShellTool::sandboxed`]). [`default_toolkit`] uses trusted shell; swap in
//! [`ShellTool::sandboxed`] with a [`ovo_sandbox::SandboxBackend`] for
//! process isolation (e.g. feature `seatbelt` → `SeatbeltBackend` on macOS).

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
use ovo_sandbox::TrustedExecution;
use ovo_tools::SharedTool;
pub use path_util::{PathJailError, resolve_jailed};
pub use read_file::ReadFileTool;
pub use shell::ShellTool;
pub use write_file::WriteFileTool;

/// Convenience bundle: read / write / grep / glob / **trusted** shell.
///
/// Shell is constructed with [`TrustedExecution`] (explicit opt-out of process
/// sandbox). Replace shell with [`ShellTool::sandboxed`] for production.
#[must_use]
pub fn default_toolkit(jail: impl Into<PathBuf>) -> Vec<SharedTool> {
    let root = jail.into();
    vec![
        Arc::new(ReadFileTool::with_jail(root.clone())),
        Arc::new(WriteFileTool::with_jail(root.clone())),
        Arc::new(GrepTool::with_jail(root.clone())),
        Arc::new(GlobTool::with_jail(root.clone())),
        Arc::new(ShellTool::trusted(root, TrustedExecution)),
    ]
}
