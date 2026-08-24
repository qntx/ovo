//! Model-driven dynamic delegation: parent agent calls `spawn_agent` tool.
//!
//! Parent-only `SpawnAgentTool`: host tools stay empty (no second `InProcessHost`).
//! Spawn metadata is not Destructive, so `TurnOptions::default` is fine.
//! 非 S-plane：AutoApprove。生产用 `TurnOptions::for_host` + `sandboxed_host`.
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    unused_crate_dependencies,
    reason = "offline demo binary uses stdout and expect for setup"
)]

use std::sync::Arc;

use ovo::{
    AgentBuilder, ConversationState, InProcessHost, LlmSampler, Message, MockSampler, SessionHost,
    SpawnAgentTool, ToolCall, ToolCallId, TurnInput, TurnOptions, TurnRuntime,
    VecConversationState,
};
use serde_json::json;

#[tokio::main]
async fn main() {
    // Child sampler responses keyed by prompt (children share the sampler).
    let sampler: Arc<MockSampler> = Arc::new(MockSampler::new());
    sampler.map_user_text("Research alpha", "alpha-report");
    sampler.map_user_text("Research beta", "beta-report");
    // Parent first samples: request two spawn_agent tool calls, then finalize.
    let id1 = ToolCallId::new("call_1").expect("id");
    let id2 = ToolCallId::new("call_2").expect("id");
    sampler.push_tools(Message::assistant_tools(vec![
        ToolCall {
            id: id1,
            name: "spawn_agent".into(),
            arguments: json!({
                "prompt": "Research alpha",
                "label": "worker-alpha"
            }),
        },
        ToolCall {
            id: id2,
            name: "spawn_agent".into(),
            arguments: json!({
                "prompt": "Research beta",
                "label": "worker-beta"
            }),
        },
    ]));
    sampler.push_text("Both workers finished; synthesis complete.");

    let sampler_dyn: Arc<dyn LlmSampler> = sampler.clone();
    let host: Arc<dyn SessionHost> =
        Arc::new(InProcessHost::new(Arc::clone(&sampler_dyn), Vec::new()).with_agent_budget(8));
    let spawn_tool = Arc::new(SpawnAgentTool::new(host));

    let parent = AgentBuilder::named("orchestrator")
        .instructions("Delegate subtasks with spawn_agent, then summarize.")
        .model("mock")
        .tools(vec![spawn_tool])
        .build()
        .expect("parent");

    let mut state = VecConversationState::new();
    let outcome = TurnRuntime::new()
        .run(
            &parent,
            sampler_dyn.as_ref(),
            &mut state,
            TurnInput::Text("Investigate alpha and beta.".into()),
            TurnOptions::default(),
        )
        .await
        .expect("parent turn");

    println!("parent_output={}", outcome.output_text);
    println!("parent_steps={}", outcome.steps);
    let tool_msgs: Vec<_> = state
        .messages()
        .iter()
        .filter(|m| m.role == ovo::Role::Tool)
        .collect();
    println!("tool_results={}", tool_msgs.len());
    for (i, m) in tool_msgs.iter().enumerate() {
        println!("tool_{i}={}", m.text());
    }
}
