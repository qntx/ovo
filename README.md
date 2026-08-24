<!-- markdownlint-disable MD033 MD041 MD036 -->

# Ovo

[![Crates.io](https://img.shields.io/crates/v/ovo.svg)](https://crates.io/crates/ovo)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

**Embeddable multi-agent runtime kernel** for Rust (dual-mode: dynamic spawn + journaled Rhai workflow).

Formerly published as [`machi`](https://crates.io/crates/machi).

**0.9.x is pre-stability.** The GitHub non-prerelease `0.9.1` is not 1.0. Breaking changes land without compatibility layers.

## Embed

Production path: `TurnOptions::for_host(gate)` + `sandboxed_host` /
`default_toolkit(jail, backend)` (canonicalizes `jail`) + the OS helper
contract. `InProcessHost::new` defaults to `AlwaysDeny`; the embedder wires
approval UI. `TurnOptions::default()` is `AutoApprove` for tests / offline
`TurnRuntime` only.

`full` is not an OS sandbox. `--all-features` is not `full`. **`runtime` is
not `toolkit`:** default `cargo add ovo` does not compile toolkit; it does
link `ovo-sandbox` via `TrustedExecution`.

Linux Landlock: ship `ovo-landlock` in the same directory as the host binary,
set `OVO_LANDLOCK_HELPER`, or put it on `PATH`
(`cargo install ovo-sandbox --features landlock --bin ovo-landlock`).
`cargo add ovo` does not install the helper.

`platform_sandbox()` can return `Failed` for a pure library dependency
(missing `seatbelt`/`landlock`, or helper not resolved). It is not
out-of-the-box. There is no `NoSandbox` fallback.

### Constructors

| Constructor | Role |
|-------------|------|
| `TurnOptions::for_host(gate)` | Production turn: `ApprovalPolicy::Destructive` + caller gate |
| `TurnOptions::default()` | `AutoApprove`; not a production default |
| `sandboxed_host(sampler, backend, jail)` | Production host (`toolkit`): sandboxed toolkit + `ChildToolkit` rebuild |
| `InProcessHost::new` | Budget 128, `AlwaysDeny`; does not install an OS backend |
| `default_toolkit(jail, backend)` | Canonicalize jail; sandboxed shell + `workspace_shell()` |
| `trusted_toolkit(jail, TrustedExecution)` | Explicit opt-out of process isolation |
| `platform_sandbox()` | Target/feature OS backend; **can Fail** |

```rust
use std::sync::Arc;
use ovo::{AlwaysDeny, TurnOptions, sandboxed_host};

// `backend` is `Arc<dyn SandboxBackend>`: `SeatbeltBackend` (macOS,
// feature `seatbelt`) or `LandlockBackend::{new, with_helper}` (Linux,
// feature `landlock`). Do not treat `platform_sandbox()` as turnkey.
let host = sandboxed_host(sampler, backend, jail)?;
let opts = TurnOptions::for_host(Arc::new(AlwaysDeny)); // replace with UI gate
```

### Features

| Feature | Compiles | OS sandbox |
|---------|----------|------------|
| default (`runtime` + `workflow`) | kernel + workflow; links `ovo-sandbox` for `TrustedExecution` | no |
| `toolkit` | cwd-jailed fs/shell | no (needs `seatbelt` / `landlock`) |
| `sandbox` | policy types / `SandboxBackend` | no |
| `seatbelt` | macOS `/usr/bin/sandbox-exec` | macOS |
| `landlock` | Linux `ovo-landlock` helper | Linux (helper must exist) |
| `full` | toolkit + state + obs + openai/ollama + sandbox types | **no** |
| `--all-features` | `full` plus `seatbelt` and `landlock` | compile-time backends; Linux still needs the helper |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.

---

<div align="center">

A **[QuantX](https://qntx.org)** open-source project.

<a href="https://qntx.org"><img alt="QuantX" width="369" src="https://raw.githubusercontent.com/qntx/.github/main/profile/qntx.svg" /></a>

Code is law. We write both.

</div>
