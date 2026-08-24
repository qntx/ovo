//! Dependency firewall: workflow must not depend on llm/http.
#![allow(
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::tests_outside_test_module,
    reason = "integration asserts; test binary links crate deps"
)]

use std::fs;
use std::path::PathBuf;

#[test]
fn workflow_cargo_toml_has_no_llm_or_http() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workflow = root.join("..").join("ovo-workflow").join("Cargo.toml");
    let text = fs::read_to_string(&workflow).expect("read workflow Cargo.toml");
    for forbidden in ["ovo-llm", "ovo-runtime", "reqwest", "hyper", "ureq"] {
        assert!(
            !text.contains(forbidden),
            "ovo-workflow must not depend on {forbidden}"
        );
    }
    assert!(text.contains("rhai"), "ovo-workflow should use rhai");
}

#[test]
fn runtime_dependencies_exclude_toolkit() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text = fs::read_to_string(root.join("Cargo.toml")).expect("runtime Cargo.toml");
    let deps = text.split("[dev-dependencies]").next().expect("deps");
    assert!(
        !deps.contains("ovo-toolkit"),
        "ovo-runtime [dependencies] must not include ovo-toolkit"
    );
    assert!(
        deps.contains("ovo-sandbox"),
        "ovo-runtime [dependencies] includes ovo-sandbox for TrustedExecution"
    );
}

#[test]
fn types_cargo_toml_stays_pure() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let types = root.join("..").join("ovo-types").join("Cargo.toml");
    let text = fs::read_to_string(&types).expect("read types Cargo.toml");
    for forbidden in ["reqwest", "ovo-llm", "ovo-runtime", "tokio"] {
        assert!(
            !text.contains(forbidden),
            "ovo-types must not depend on {forbidden}"
        );
    }
}
