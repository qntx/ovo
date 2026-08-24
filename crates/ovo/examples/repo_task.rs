//! Agent + jailed toolkit writes a file (offline mock).
//!
//! 非 S-plane：AutoApprove + trusted shell。生产用 `TurnOptions::for_host` + `sandboxed_host`.
//!
//! ```bash
//! cargo run -p ovo --example repo_task --features toolkit
//! ```
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    unused_crate_dependencies,
    reason = "demo binary"
)]

use std::sync::Arc;

use ovo::{
    AgentBuilder, Message, MockSampler, PrometheusRecorder, SharedMetrics, ToolCall, ToolCallId,
    TrustedExecution, TurnInput, TurnOptions, TurnRuntime, VecConversationState, trusted_toolkit,
};
use serde_json::json;
use tempfile::tempdir;

#[tokio::main]
async fn main() {
    let dir = tempdir().expect("tmp workspace");
    let jail = dir.path().to_path_buf();
    println!("workspace={}", jail.display());

    let write_id = ToolCallId::new("w1").expect("id");
    let sampler = Arc::new(MockSampler::new());
    sampler.push_tools(Message::assistant_tools(vec![ToolCall {
        id: write_id,
        name: "write_file".into(),
        arguments: json!({
            "path": "hello.txt",
            "content": "ovo-n1-vertical-slice\n"
        }),
    }]));
    sampler.push_text("Wrote hello.txt under the workspace jail.");

    let tools = trusted_toolkit(&jail, TrustedExecution);
    let agent = AgentBuilder::named("coder")
        .instructions("Use write_file to create files. Keep answers short.")
        .model("mock")
        .tools(tools)
        .max_steps(8)
        .build()
        .expect("agent");

    let metrics = Arc::new(PrometheusRecorder::new());
    let metrics_dyn: SharedMetrics = metrics.clone();
    let mut state = VecConversationState::new();
    let out = TurnRuntime::new()
        .run(
            &agent,
            sampler.as_ref(),
            &mut state,
            TurnInput::Text("Create hello.txt with a short marker line.".into()),
            TurnOptions {
                cwd: Some(jail.clone()),
                max_steps: Some(8),
                metrics: metrics_dyn,
                ..TurnOptions::default()
            },
        )
        .await
        .expect("turn");

    let path = jail.join("hello.txt");
    let body = std::fs::read_to_string(&path).expect("read written file");
    println!("output={}", out.output_text);
    println!("steps={}", out.steps);
    println!("file={}", path.display());
    println!("file_body={body:?}");
    assert!(
        body.contains("ovo-n1-vertical-slice"),
        "expected toolkit write side effect"
    );

    let prom = metrics.render();
    assert!(
        prom.contains("ovo_turns_total") || prom.contains("ovo_tool_calls_total"),
        "metrics export empty:\n{prom}"
    );
    println!(
        "metrics_excerpt={}",
        prom.lines().take(8).collect::<Vec<_>>().join(" | ")
    );
    println!("repo_task_ok=true");
}
