//! Landlock helper: apply policy then exec. Linux-only enforcement.

#![forbid(unsafe_code)]
#![allow(
    unused_crate_dependencies,
    reason = "bin uses a subset of package deps"
)]

use std::process::ExitCode;

#[cfg(target_os = "linux")]
#[allow(clippy::print_stderr, reason = "helper diagnostics before exec")]
fn main() -> ExitCode {
    linux_main()
}

#[cfg(not(target_os = "linux"))]
#[allow(clippy::print_stderr, reason = "helper reports OS mismatch on stderr")]
fn main() -> ExitCode {
    eprintln!("ovo-landlock: Linux only");
    ExitCode::from(1)
}

#[cfg(target_os = "linux")]
#[allow(clippy::print_stderr, reason = "helper diagnostics before exec")]
fn linux_main() -> ExitCode {
    use std::os::unix::process::CommandExt;

    use ovo_sandbox::SandboxPolicy;

    let mut args = std::env::args_os();
    let _argv0 = args.next();
    let Some(flag) = args.next() else {
        return usage();
    };
    if flag != "-p" {
        return usage();
    }
    let Some(json) = args.next() else {
        return usage();
    };
    let Some(program) = args.next() else {
        return usage();
    };
    let Some(json) = json.to_str() else {
        eprintln!("ovo-landlock: policy JSON must be UTF-8");
        return ExitCode::from(2);
    };
    let policy: SandboxPolicy = match serde_json::from_str(json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ovo-landlock: invalid policy JSON: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = ovo_sandbox::apply_landlock_policy(&policy) {
        eprintln!("ovo-landlock: {e}");
        return ExitCode::from(1);
    }
    let mut cmd = std::process::Command::new(program);
    cmd.args(args);
    let err = cmd.exec();
    eprintln!("ovo-landlock: exec: {err}");
    ExitCode::from(1)
}

#[cfg(target_os = "linux")]
#[allow(clippy::print_stderr, reason = "helper diagnostics before exec")]
fn usage() -> ExitCode {
    eprintln!("ovo-landlock: usage: ovo-landlock -p <policy-json> <program> [args...]");
    ExitCode::from(2)
}
