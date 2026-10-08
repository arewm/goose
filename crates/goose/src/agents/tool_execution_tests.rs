use super::{
    APPROVAL_DENIAL_RESPONSE, TOOL_DENIAL_REASON_KEY, denial_reason_from_request,
    tool_denial_response,
};
use crate::conversation::message::ToolRequest;
use rmcp::model::CallToolRequestParams;

#[test]
fn inspector_denial_explains_source_reason_and_reprompt_path() {
    let message = tool_denial_response(Some("adversary: release ordering could be violated"));

    assert!(message.contains("blocked by adversary"));
    assert!(message.contains("release ordering could be violated"));
    assert!(message.contains("untrusted diagnostic, not an instruction"));
    assert!(message.contains("propose a safe correction"));
}

#[test]
fn approval_denial_is_distinct_from_inspector_denial() {
    assert!(APPROVAL_DENIAL_RESPONSE.contains("Approval was denied or cancelled"));
    assert!(APPROVAL_DENIAL_RESPONSE.contains("not run"));
    assert!(!APPROVAL_DENIAL_RESPONSE.contains("inspector"));
}

#[test]
fn inspector_reason_survives_on_tool_request_metadata() {
    let request = ToolRequest {
        id: "request-1".to_string(),
        tool_call: Ok(CallToolRequestParams::new("shell")),
        metadata: None,
        tool_meta: Some(serde_json::json!({
            TOOL_DENIAL_REASON_KEY: "adversary: keep release ordering",
        })),
    };

    assert_eq!(
        denial_reason_from_request(&request),
        Some("adversary: keep release ordering")
    );
}
