# Ovo

Enterprise embeddable **agent runtime kernel** for Rust.

Formerly published as [`machi`](https://crates.io/crates/machi).

**0.9.x is pre-stability.** GitHub non-prerelease 0.9.1 is not 1.0.
Production: `TurnOptions::for_host(gate)` + `sandboxed_host` /
`default_toolkit(jail, backend)` + an installed OS backend. Linux helper:
same directory as the host binary, `OVO_LANDLOCK_HELPER`, or PATH;
`cargo add ovo` does not compile toolkit and does not install `ovo-landlock`.
Host default approval is `AlwaysDeny`; the embedder wires the UI.
`platform_sandbox()` can Fail — not turnkey for a pure library dependency.
