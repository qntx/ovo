//! Landlock wrap integration tests (Linux + feature `landlock`).
#![allow(
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::tests_outside_test_module,
    reason = "integration test binary; linux-only body"
)]

#[cfg(all(feature = "landlock", target_os = "linux"))]
mod linux {
    use std::path::{Path, PathBuf};
    use std::process::Stdio;

    use landlock::{AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr};
    use ovo_sandbox::{FsPolicy, LandlockBackend, NetPolicy, SandboxBackend, SandboxPolicy};
    use tokio::process::Command;

    fn helper() -> LandlockBackend {
        LandlockBackend::with_helper(env!("CARGO_BIN_EXE_ovo-landlock"))
    }

    fn fs_rw_net_allowed(root: PathBuf) -> SandboxPolicy {
        SandboxPolicy {
            fs: FsPolicy::ReadWrite { paths: vec![root] },
            net: NetPolicy::Allowed,
        }
    }

    fn jail_root() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("temp");
        let root = dir.path().canonicalize().expect("canon");
        (dir, root)
    }

    fn kernel_handles_truncate() -> bool {
        Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::Truncate)
            .is_ok()
    }

    fn posix_single_quote(path: &Path) -> String {
        format!("'{}'", path.to_string_lossy().replace('\'', r#"'"'"'"#))
    }

    #[tokio::test]
    async fn inside_read_succeeds() {
        let (_dir, root) = jail_root();
        std::fs::write(root.join("in.txt"), b"inside-ok").expect("in");
        let policy = fs_rw_net_allowed(root.clone());
        let mut cmd = Command::new("/bin/cat");
        cmd.arg(root.join("in.txt"));
        let mut cmd = helper().wrap(&policy, cmd).expect("wrap");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd.output().await.expect("exec");
        assert!(
            output.status.success(),
            "inside read must succeed under FS Landlock, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("inside-ok"),
            "stdout={:?}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[tokio::test]
    async fn home_outside_read_fails() {
        let (_dir, root) = jail_root();
        std::fs::write(root.join("in.txt"), b"inside-ok").expect("in");
        let outside = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME")
            .join(format!("ovo_ll_out_{}", std::process::id()));
        std::fs::write(&outside, b"secret").expect("out");
        let policy = fs_rw_net_allowed(root);
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(format!("cat {}", outside.display()));
        let mut cmd = helper().wrap(&policy, cmd).expect("wrap");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd.output().await.expect("exec");
        let _ = std::fs::remove_file(&outside);
        assert!(
            !output.status.success(),
            "outside read must fail under landlock, stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[tokio::test]
    async fn fs_not_enforced_is_failed() {
        let (_dir, root) = jail_root();
        let policy = fs_rw_net_allowed(root);
        let cmd = Command::new("/bin/true");
        let mut cmd = helper().wrap(&policy, cmd).expect("wrap");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd.output().await.expect("exec");
        assert!(
            output.status.success(),
            "FS Landlock must be enforced, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[tokio::test]
    async fn net_denied_is_separate() {
        let (_dir, root) = jail_root();
        let policy = SandboxPolicy {
            fs: FsPolicy::ReadWrite {
                paths: vec![root.clone()],
            },
            net: NetPolicy::Denied,
        };
        let probe = Command::new("/bin/true");
        let mut probe = helper().wrap(&policy, probe).expect("wrap");
        probe
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let probe_out = probe.output().await.expect("probe");
        if !probe_out.status.success() {
            return;
        }
        if !Path::new("/usr/bin/curl").is_file() {
            return;
        }
        let mut curl = Command::new("/usr/bin/curl");
        curl.args(["-sS", "--max-time", "2", "https://example.com"]);
        let mut curl = helper().wrap(&policy, curl).expect("wrap curl");
        curl.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let out = curl.output().await.expect("curl");
        assert!(
            !out.status.success(),
            "network must be denied, stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[tokio::test]
    async fn stdio_left_to_caller() {
        let (_dir, root) = jail_root();
        let policy = fs_rw_net_allowed(root);
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("landlock-stdio");
        let mut cmd = helper().wrap(&policy, cmd).expect("wrap");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd.output().await.expect("exec");
        assert!(
            output.status.success(),
            "echo must succeed, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("landlock-stdio"),
            "stdout={:?}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[tokio::test]
    async fn truncate_outside_jail_denied() {
        if !kernel_handles_truncate() || !Path::new("/usr/bin/python3").is_file() {
            return;
        }
        let (_dir, root) = jail_root();
        let outside = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME")
            .join(format!("ovo_ll_trunc_{}", std::process::id()));
        std::fs::write(&outside, b"secret-data").expect("out");
        let policy = fs_rw_net_allowed(root);
        let mut cmd = Command::new("python3");
        cmd.arg("-c")
            .arg("import os,sys; os.truncate(sys.argv[1], 0)")
            .arg(&outside);
        let mut cmd = helper().wrap(&policy, cmd).expect("wrap");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd.output().await.expect("exec");
        let leftover = std::fs::read(&outside).expect("leftover");
        let _ = std::fs::remove_file(&outside);
        assert!(
            !output.status.success(),
            "outside truncate must fail under landlock, stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            leftover, b"secret-data",
            "outside file must stay secret-data"
        );
    }

    #[tokio::test]
    async fn truncate_inside_jail_succeeds() {
        let (_dir, root) = jail_root();
        let inside = root.join("in.txt");
        std::fs::write(&inside, b"inside-data").expect("in");
        let policy = fs_rw_net_allowed(root);
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(format!(": > {}", posix_single_quote(&inside)));
        let mut cmd = helper().wrap(&policy, cmd).expect("wrap");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd.output().await.expect("exec");
        assert!(
            output.status.success(),
            "inside truncate must succeed under FS Landlock, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        let leftover = std::fs::read(&inside).expect("leftover");
        assert_eq!(leftover, b"", "inside file must be empty");
    }
}
