//! macOS Seatbelt backend via `/usr/bin/sandbox-exec`.
//!
//! Enabled with feature `seatbelt` on macOS only.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use tokio::process::Command;

use crate::{
    FsPolicy, NetPolicy, SandboxBackend, SandboxError, SandboxPolicy, require_absolute,
    take_command_spec,
};

/// Absolute path to the system seatbelt helper (do not honor PATH).
pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Seatbelt (`sandbox-exec`) backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct SeatbeltBackend;

impl SandboxBackend for SeatbeltBackend {
    fn name(&self) -> &'static str {
        "seatbelt"
    }

    fn wrap(&self, policy: &SandboxPolicy, cmd: Command) -> Result<Command, SandboxError> {
        if !Path::new(SANDBOX_EXEC).is_file() {
            return Err(SandboxError::Failed(format!("{SANDBOX_EXEC} not found")));
        }
        let profile = build_profile(policy)?;
        rewrite_with_seatbelt(cmd, &profile)
    }
}

/// Build a minimal SBPL string from [`SandboxPolicy`].
///
/// # Errors
///
/// Returns [`SandboxError::Denied`] when a policy path is not absolute.
pub fn build_profile(policy: &SandboxPolicy) -> Result<String, SandboxError> {
    let mut paths_ro: Vec<PathBuf> = Vec::new();
    let mut paths_rw: Vec<PathBuf> = Vec::new();
    match &policy.fs {
        FsPolicy::None => {}
        FsPolicy::ReadOnly { paths } => {
            for p in paths {
                paths_ro.push(require_absolute(p)?);
            }
        }
        FsPolicy::ReadWrite { paths } => {
            for p in paths {
                paths_rw.push(require_absolute(p)?);
            }
        }
    }

    let mut out = String::from(
        r#"(version 1)
(deny default)
(allow process*)
(allow signal)
(allow sysctl*)
(allow mach*)
(allow iokit-open)
(allow file-ioctl)
(allow file-read-metadata)
(allow file-read*
  (subpath "/usr")
  (subpath "/bin")
  (subpath "/sbin")
  (subpath "/System")
  (subpath "/Library")
  (subpath "/Applications")
  (subpath "/private/var/db")
  (subpath "/private/var/folders")
  (subpath "/private/var/select")
  (subpath "/private/etc")
  (subpath "/etc")
  (subpath "/dev")
  (subpath "/tmp")
  (subpath "/private/tmp")
  (subpath "/var")
  (literal "/")
  (literal "/dev/null")
"#,
    );
    for p in paths_ro.iter().chain(paths_rw.iter()) {
        let _ = writeln!(out, "  (subpath \"{}\")", sbpl_escape(p));
    }
    out.push_str(")\n");

    out.push_str("(allow file-write*\n  (literal \"/dev/null\")\n");
    // dyld / temp scratch often needed under user cache trees.
    out.push_str("  (subpath \"/private/var/folders\")\n");
    for p in &paths_rw {
        let _ = writeln!(out, "  (subpath \"{}\")", sbpl_escape(p));
    }
    out.push_str(")\n");

    if matches!(policy.net, NetPolicy::Allowed) {
        out.push_str("(allow network*)\n");
    }

    Ok(out)
}

fn sbpl_escape(p: &Path) -> String {
    p.to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn rewrite_with_seatbelt(cmd: Command, profile: &str) -> Result<Command, SandboxError> {
    let spec = take_command_spec(&cmd);
    let mut wrapped = Command::new(SANDBOX_EXEC);
    wrapped.arg("-p").arg(profile);
    wrapped.arg(&spec.program);
    wrapped.args(spec.args);
    if let Some(dir) = spec.cwd {
        wrapped.current_dir(dir);
    }
    for (k, v) in spec.envs {
        wrapped.env(k, v);
    }
    wrapped.kill_on_drop(true);
    Ok(wrapped)
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;
    use crate::SandboxPolicy;

    #[test]
    fn profile_includes_workspace_and_denies_network_by_default() {
        let p = SandboxPolicy::workspace("/tmp/ws");
        let prof = build_profile(&p).expect("profile");
        assert!(prof.contains("(subpath \"/tmp/ws\")"), "{prof}");
        assert!(!prof.contains("(allow network*)"), "{prof}");
        assert!(prof.contains("(deny default)"), "{prof}");
    }

    #[test]
    fn profile_allows_network_when_requested() {
        let mut p = SandboxPolicy::workspace("/tmp/ws");
        p.net = NetPolicy::Allowed;
        let prof = build_profile(&p).expect("profile");
        assert!(prof.contains("(allow network*)"), "{prof}");
    }

    #[test]
    fn rejects_relative_paths() {
        let p = SandboxPolicy {
            fs: FsPolicy::ReadWrite {
                paths: vec![PathBuf::from("relative")],
            },
            net: NetPolicy::Denied,
        };
        let err = build_profile(&p).expect_err("relative");
        assert!(matches!(err, SandboxError::Denied(_)));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn wrap_prefixes_sandbox_exec() {
        let policy = SandboxPolicy::workspace("/tmp/ws");
        let cmd = Command::new("/bin/echo");
        let wrapped = SeatbeltBackend.wrap(&policy, cmd).expect("wrap");
        assert_eq!(wrapped.as_std().get_program(), SANDBOX_EXEC);
        let args: Vec<_> = wrapped.as_std().get_args().collect();
        let flag = args.first().expect("flag");
        let profile = args.get(1).expect("profile");
        let prog = args.get(2).expect("program");
        assert_eq!(*flag, std::ffi::OsStr::new("-p"));
        assert!(
            profile.to_string_lossy().contains("(deny default)"),
            "{profile:?}"
        );
        assert_eq!(*prog, std::ffi::OsStr::new("/bin/echo"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn seatbelt_blocks_outside_read() {
        use std::io::Write;
        use std::process::Command as StdCommand;

        let root = scratch_dir("ws");
        std::fs::write(root.join("in.txt"), b"inside").expect("in");
        // Prefer HOME: /tmp trees are allowed for dyld/scratch in the profile.
        let outside = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(format!("ovo_out_{}", std::process::id()));
        {
            let mut f = std::fs::File::create(&outside).expect("create out");
            f.write_all(b"secret").expect("write");
        }

        let policy = SandboxPolicy::workspace(&root);
        let profile = build_profile(&policy).expect("profile");

        let status_in = StdCommand::new(SANDBOX_EXEC)
            .arg("-p")
            .arg(&profile)
            .arg("/bin/cat")
            .arg(root.join("in.txt"))
            .status()
            .expect("run in");
        assert!(status_in.success(), "inside read must succeed");

        let out = StdCommand::new(SANDBOX_EXEC)
            .arg("-p")
            .arg(&profile)
            .arg("/bin/cat")
            .arg(&outside)
            .output()
            .expect("run out");
        assert!(
            !out.status.success(),
            "outside read must fail, stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn seatbelt_blocks_network_when_denied() {
        use std::process::Command as StdCommand;

        let root = scratch_dir("net");
        let policy = SandboxPolicy::workspace(&root);
        assert_eq!(policy.net, NetPolicy::Denied);
        let profile = build_profile(&policy).expect("profile");

        let curl = Path::new("/usr/bin/curl");
        if !curl.is_file() {
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        let out = StdCommand::new(SANDBOX_EXEC)
            .arg("-p")
            .arg(&profile)
            .arg(curl)
            .args(["-sS", "--max-time", "2", "https://example.com"])
            .output()
            .expect("run curl");
        assert!(
            !out.status.success(),
            "network must be denied under default workspace policy, stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Unique abs path under the process temp dir (no extra dev-deps).
    #[cfg(target_os = "macos")]
    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("ovo_sb_{tag}_{}_{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir.canonicalize().expect("canon")
    }
}
