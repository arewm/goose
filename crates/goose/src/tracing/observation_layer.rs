use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{span, Event, Id, Level, Metadata, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SpanData {
    pub observation_id: String, // Langfuse requires ids to be UUID v4 strings
    pub name: String,
    pub start_time: String,
    pub level: String,
    pub metadata: serde_json::Map<String, Value>,
    pub parent_span_id: Option<u64>,
}

pub fn map_level(level: &Level) -> &'static str {
    match *level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARNING",
        Level::INFO => "DEFAULT",
        Level::DEBUG => "DEBUG",
        Level::TRACE => "DEBUG",
    }
}

pub fn flatten_metadata(
    metadata: serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    let mut flattened = serde_json::Map::new();
    for (key, value) in metadata {
        match value {
            Value::String(s) => {
                flattened.insert(key, json!(s));
            }
            Value::Object(mut obj) => {
                if let Some(text) = obj.remove("text") {
                    flattened.insert(key, text);
                } else {
                    flattened.insert(key, json!(obj));
                }
            }
            _ => {
                flattened.insert(key, value);
            }
        }
    }
    flattened
}

pub trait BatchManager: Send + Sync + 'static {
    fn add_event(&mut self, event_type: &str, body: Value);
    fn send(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    fn is_empty(&self) -> bool;
}

#[derive(Debug)]
pub struct SpanTracker {
    active_spans: HashMap<u64, String>, // span_id -> observation_id. span_id in Tracing is u64 whereas Langfuse requires UUID v4 strings
    current_trace_id: Option<String>,
}

impl Default for SpanTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SpanTracker {
    pub fn new() -> Self {
        Self {
            active_spans: HashMap::new(),
            current_trace_id: None,
        }
    }

    pub fn add_span(&mut self, span_id: u64, observation_id: String) {
        self.active_spans.insert(span_id, observation_id);
    }

    pub fn get_span(&self, span_id: u64) -> Option<&String> {
        self.active_spans.get(&span_id)
    }

    pub fn remove_span(&mut self, span_id: u64) -> Option<String> {
        self.active_spans.remove(&span_id)
    }
}

#[derive(Clone)]
pub struct ObservationLayer {
    pub batch_manager: Arc<Mutex<dyn BatchManager>>,
    pub span_tracker: Arc<Mutex<SpanTracker>>,
}

impl ObservationLayer {
    pub fn handle_span(&self, span_id: u64, span_data: SpanData) {
        let observation_id = span_data.observation_id.clone();

        // Consolidate span addition, parent lookup, and trace id resolution into a single lock acquisition
        let (parent_id, trace_id, need_trace_create) = {
            let mut spans = self.span_tracker.lock().unwrap_or_else(|e| e.into_inner());
            spans.add_span(span_id, observation_id.clone());

            let parent_id = span_data
                .parent_span_id
                .and_then(|parent_span_id| spans.get_span(parent_span_id).cloned());

            if let Some(id) = spans.current_trace_id.clone() {
                (parent_id, id, false)
            } else {
                let id = Uuid::new_v4().to_string();
                spans.current_trace_id = Some(id.clone());
                (parent_id, id, true)
            }
        };

        let mut batch = self.batch_manager.lock().unwrap_or_else(|e| e.into_inner());
        if need_trace_create {
            batch.add_event(
                "trace-create",
                json!({
                    "id": trace_id,
                    "name": Utc::now().timestamp().to_string(),
                    "timestamp": Utc::now().to_rfc3339(),
                    "input": {},
                    "metadata": {},
                    "tags": [],
                    "public": false
                }),
            );
        }

        // Create the span observation
        batch.add_event(
            "observation-create",
            json!({
                "id": observation_id,
                "traceId": trace_id,
                "type": "SPAN",
                "name": span_data.name,
                "startTime": span_data.start_time,
                "parentObservationId": parent_id,
                "metadata": span_data.metadata,
                "level": span_data.level
            }),
        );
    }

    pub fn handle_span_close(&self, span_id: u64) {
        let observation_id = {
            let mut spans = self.span_tracker.lock().unwrap_or_else(|e| e.into_inner());
            spans.remove_span(span_id)
        };

        if let Some(observation_id) = observation_id {
            let trace_id = self.ensure_trace_id();
            let mut batch = self.batch_manager.lock().unwrap_or_else(|e| e.into_inner());
            batch.add_event(
                "observation-update",
                json!({
                    "id": observation_id,
                    "type": "SPAN",
                    "traceId": trace_id,
                    "endTime": Utc::now().to_rfc3339()
                }),
            );
        }
    }

    pub fn ensure_trace_id(&self) -> String {
        let (trace_id, need_trace_create) = {
            let mut spans = self.span_tracker.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(id) = spans.current_trace_id.clone() {
                (id, false)
            } else {
                let id = Uuid::new_v4().to_string();
                spans.current_trace_id = Some(id.clone());
                (id, true)
            }
        };

        if need_trace_create {
            let mut batch = self.batch_manager.lock().unwrap_or_else(|e| e.into_inner());
            batch.add_event(
                "trace-create",
                json!({
                    "id": trace_id,
                    "name": Utc::now().timestamp().to_string(),
                    "timestamp": Utc::now().to_rfc3339(),
                    "input": {},
                    "metadata": {},
                    "tags": [],
                    "public": false
                }),
            );
        }

        trace_id
    }

    pub fn update_trace(&self, updates: serde_json::Map<String, Value>) {
        let trace_id = self.ensure_trace_id();
        let mut body = json!({ "id": trace_id });
        for (k, v) in updates {
            body[k] = v;
        }
        let mut batch = self.batch_manager.lock().unwrap_or_else(|e| e.into_inner());
        batch.add_event("trace-create", body);
    }

    pub fn handle_record(&self, span_id: u64, metadata: serde_json::Map<String, Value>) {
        // Handle trace-level fields by updating the trace itself
        let trace_fields: Vec<&str> = vec!["trace_input", "trace_output"];
        let has_trace_fields = trace_fields.iter().any(|f| metadata.contains_key(*f));

        if has_trace_fields {
            let mut trace_updates = serde_json::Map::new();
            if let Some(val) = metadata.get("trace_input") {
                trace_updates.insert("input".to_string(), val.clone());
            }
            if let Some(val) = metadata.get("trace_output") {
                trace_updates.insert("output".to_string(), val.clone());
            }
            if !trace_updates.is_empty() {
                self.update_trace(trace_updates);
            }
        }

        // Filter out trace-level fields from span metadata
        let span_metadata: serde_json::Map<String, Value> = metadata
            .into_iter()
            .filter(|(k, _)| !trace_fields.contains(&k.as_str()))
            .collect();

        if span_metadata.is_empty() {
            return;
        }

        let observation_id = {
            let spans = self.span_tracker.lock().unwrap_or_else(|e| e.into_inner());
            spans.get_span(span_id).cloned()
        };

        if let Some(observation_id) = observation_id {
            let trace_id = self.ensure_trace_id();

            let mut update = json!({
                "id": observation_id,
                "traceId": trace_id,
                "type": "SPAN"
            });

            // Handle special fields
            if let Some(val) = span_metadata.get("input") {
                update["input"] = val.clone();
            }

            if let Some(val) = span_metadata.get("output") {
                update["output"] = val.clone();
            }

            if let Some(val) = span_metadata.get("model_config") {
                update["metadata"] = json!({ "model_config": val });
            }

            // Handle any remaining metadata
            let remaining_metadata: serde_json::Map<String, Value> = span_metadata
                .iter()
                .filter(|(k, _)| !["input", "output", "model_config"].contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            if !remaining_metadata.is_empty() {
                let flattened = flatten_metadata(remaining_metadata);
                if update.get("metadata").is_some() {
                    if let Some(obj) = update["metadata"].as_object_mut() {
                        for (k, v) in flattened {
                            obj.insert(k, v);
                        }
                    }
                } else {
                    update["metadata"] = json!(flattened);
                }
            }

            let mut batch = self.batch_manager.lock().unwrap_or_else(|e| e.into_inner());
            batch.add_event("span-update", update);
        }
    }
}

impl<S> Layer<S> for ObservationLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        metadata.target().starts_with("goose::")
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let span_id = id.into_u64();

        let parent_span_id = ctx
            .span_scope(id)
            .and_then(|mut scope| scope.nth(1))
            .map(|parent| parent.id().into_u64());

        let mut visitor = JsonVisitor::new();
        attrs.record(&mut visitor);

        let span_data = SpanData {
            observation_id: Uuid::new_v4().to_string(),
            name: attrs.metadata().name().to_string(),
            start_time: Utc::now().to_rfc3339(),
            level: map_level(attrs.metadata().level()).to_owned(),
            metadata: visitor.recorded_fields,
            parent_span_id,
        };

        self.handle_span(span_id, span_data);
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        let span_id = id.into_u64();
        self.handle_span_close(span_id);
    }

    fn on_record(&self, span: &Id, values: &span::Record<'_>, _ctx: Context<'_, S>) {
        let span_id = span.into_u64();
        let mut visitor = JsonVisitor::new();
        values.record(&mut visitor);
        let metadata = visitor.recorded_fields;

        if !metadata.is_empty() {
            self.handle_record(span_id, metadata);
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut visitor = JsonVisitor::new();
        event.record(&mut visitor);
        let metadata = visitor.recorded_fields;

        if let Some(span_id) = ctx.lookup_current().map(|span| span.id().into_u64()) {
            self.handle_record(span_id, metadata);
        }
    }
}

#[derive(Debug)]
struct JsonVisitor {
    recorded_fields: serde_json::Map<String, Value>,
}

impl JsonVisitor {
    fn new() -> Self {
        Self {
            recorded_fields: serde_json::Map::new(),
        }
    }

    fn insert_value(&mut self, field: &Field, value: Value) {
        self.recorded_fields.insert(field.name().to_string(), value);
    }
}

macro_rules! record_field {
    ($fn_name:ident, $type:ty) => {
        fn $fn_name(&mut self, field: &Field, value: $type) {
            self.insert_value(field, Value::from(value));
        }
    };
}

impl Visit for JsonVisitor {
    record_field!(record_i64, i64);
    record_field!(record_u64, u64);
    record_field!(record_bool, bool);
    record_field!(record_str, &str);

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.insert_value(field, Value::String(format!("{:?}", value)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::dispatcher;
    use tracing_subscriber::layer::SubscriberExt;

    type Events = Arc<Mutex<Vec<(String, Value)>>>;
    struct TestFixture {
        original_subscriber: Option<dispatcher::Dispatch>,
        events: Option<Events>,
    }

    impl TestFixture {
        fn new() -> Self {
            Self {
                original_subscriber: Some(dispatcher::get_default(dispatcher::Dispatch::clone)),
                events: None,
            }
        }

        fn with_test_layer(mut self) -> (Self, ObservationLayer) {
            let events = Arc::new(Mutex::new(Vec::new()));
            let mock_manager = MockBatchManager::new(events.clone());

            let layer = ObservationLayer {
                batch_manager: Arc::new(Mutex::new(mock_manager)),
                span_tracker: Arc::new(Mutex::new(SpanTracker::new())),
            };

            self.events = Some(events);
            (self, layer)
        }

        fn get_events(&self) -> Vec<(String, Value)> {
            self.events
                .as_ref()
                .expect("Events not initialized")
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    }

    impl Drop for TestFixture {
        fn drop(&mut self) {
            if let Some(subscriber) = &self.original_subscriber {
                let _ = dispatcher::set_global_default(subscriber.clone());
            }
        }
    }

    struct MockBatchManager {
        events: Arc<Mutex<Vec<(String, Value)>>>,
    }

    impl MockBatchManager {
        fn new(events: Arc<Mutex<Vec<(String, Value)>>>) -> Self {
            Self { events }
        }
    }

    impl BatchManager for MockBatchManager {
        fn add_event(&mut self, event_type: &str, body: Value) {
            self.events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((event_type.to_string(), body));
        }

        fn send(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }

        fn is_empty(&self) -> bool {
            self.events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        }
    }

    fn create_test_span_data() -> SpanData {
        SpanData {
            observation_id: Uuid::new_v4().to_string(),
            name: "test_span".to_string(),
            start_time: Utc::now().to_rfc3339(),
            level: "DEFAULT".to_string(),
            metadata: serde_json::Map::new(),
            parent_span_id: None,
        }
    }

    #[test]
    fn test_span_creation() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let span_id = 1u64;
        let span_data = create_test_span_data();

        layer.handle_span(span_id, span_data.clone());

        let events = fixture.get_events();
        assert_eq!(events.len(), 2); // trace-create and observation-create

        let (event_type, body) = &events[1];
        assert_eq!(event_type, "observation-create");
        assert_eq!(body["id"], span_data.observation_id);
        assert_eq!(body["name"], "test_span");
        assert_eq!(body["type"], "SPAN");
    }

    #[test]
    fn test_span_close() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let span_id = 1u64;
        let span_data = create_test_span_data();

        layer.handle_span(span_id, span_data.clone());
        layer.handle_span_close(span_id);

        let events = fixture.get_events();
        assert_eq!(events.len(), 3); // trace-create, observation-create, observation-update

        let (event_type, body) = &events[2];
        assert_eq!(event_type, "observation-update");
        assert_eq!(body["id"], span_data.observation_id);
        assert!(body["endTime"].as_str().is_some());
    }

    #[test]
    fn test_record_handling() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let span_id = 1u64;
        let span_data = create_test_span_data();

        layer.handle_span(span_id, span_data.clone());

        let mut metadata = serde_json::Map::new();
        metadata.insert("input".to_string(), json!("test input"));
        metadata.insert("output".to_string(), json!("test output"));
        metadata.insert("custom_field".to_string(), json!("custom value"));

        layer.handle_record(span_id, metadata);

        let events = fixture.get_events();
        assert_eq!(events.len(), 3); // trace-create, observation-create, span-update

        let (event_type, body) = &events[2];
        assert_eq!(event_type, "span-update");
        assert_eq!(body["input"], "test input");
        assert_eq!(body["output"], "test output");
        assert_eq!(body["metadata"]["custom_field"], "custom value");
    }

    #[test]
    fn test_trace_input_output_updates() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let span_id = 1u64;
        let span_data = create_test_span_data();

        layer.handle_span(span_id, span_data.clone());

        let mut metadata = serde_json::Map::new();
        metadata.insert("trace_input".to_string(), json!("hello from user"));
        metadata.insert("trace_output".to_string(), json!("response from assistant"));

        layer.handle_record(span_id, metadata);

        let events = fixture.get_events();
        // trace-create, observation-create, trace-create (update with input/output)
        assert!(events.len() >= 3);

        let trace_update = events
            .iter()
            .rfind(|(t, b)| t == "trace-create" && b.get("input").is_some_and(|v| v.is_string()))
            .expect("should have a trace update with input");
        assert_eq!(trace_update.1["input"], "hello from user");
        assert_eq!(trace_update.1["output"], "response from assistant");
    }

    #[test]
    fn test_trace_fields_not_sent_as_span_metadata() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let span_id = 1u64;
        let span_data = create_test_span_data();

        layer.handle_span(span_id, span_data.clone());

        // Only trace-level fields, no span-level fields
        let mut metadata = serde_json::Map::new();
        metadata.insert("trace_input".to_string(), json!("user msg"));

        layer.handle_record(span_id, metadata);

        let events = fixture.get_events();
        // Should NOT have a span-update since there are no span-level fields
        let span_updates: Vec<_> = events.iter().filter(|(t, _)| t == "span-update").collect();
        assert!(
            span_updates.is_empty(),
            "trace-only fields should not generate span-update events"
        );
    }

    #[test]
    fn test_mixed_trace_and_span_fields() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let span_id = 1u64;
        let span_data = create_test_span_data();

        layer.handle_span(span_id, span_data.clone());

        let mut metadata = serde_json::Map::new();
        metadata.insert("trace_input".to_string(), json!("user msg"));
        metadata.insert("input".to_string(), json!("tool input"));
        metadata.insert("output".to_string(), json!("tool output"));

        layer.handle_record(span_id, metadata);

        let events = fixture.get_events();

        // Should have both a trace update and a span-update
        let trace_updates: Vec<_> = events
            .iter()
            .filter(|(t, b)| t == "trace-create" && b.get("input").is_some_and(|v| v.is_string()))
            .collect();
        assert_eq!(trace_updates.len(), 1);
        assert_eq!(trace_updates[0].1["input"], "user msg");

        let span_updates: Vec<_> = events.iter().filter(|(t, _)| t == "span-update").collect();
        assert_eq!(span_updates.len(), 1);
        assert_eq!(span_updates[0].1["input"], "tool input");
        assert_eq!(span_updates[0].1["output"], "tool output");
    }

    #[test]
    fn test_flatten_metadata() {
        let _fixture = TestFixture::new();
        let mut metadata = serde_json::Map::new();
        metadata.insert("simple".to_string(), json!("value"));
        metadata.insert(
            "complex".to_string(),
            json!({
                "text": "inner value"
            }),
        );

        let flattened = flatten_metadata(metadata);
        assert_eq!(flattened["simple"], "value");
        assert_eq!(flattened["complex"], "inner value");
    }

    /// Verifies that ObservationLayer tracing hooks executed on non-Tokio worker threads
    /// (e.g. sqlx-sqlite worker threads, rayon threads, std::threads) do not panic or crash.
    #[test]
    fn test_non_tokio_thread_span_lifecycle() {
        let (fixture, layer) = TestFixture::new().with_test_layer();
        let subscriber = tracing_subscriber::Registry::default().with(layer);

        // Run tracing on a plain std::thread with no Tokio reactor running
        let handle = std::thread::spawn(move || {
            tracing::subscriber::with_default(subscriber, || {
                let span = tracing::info_span!(
                    target: "goose::test",
                    "worker_thread_span",
                    input = "thread_input"
                );
                let _enter = span.enter();
                tracing::info!(target: "goose::test", "event_on_worker_thread");
                drop(_enter);
                drop(span);
            });
        });

        assert!(
            handle.join().is_ok(),
            "Thread panicked during span lifecycle"
        );

        let events = fixture.get_events();
        assert!(
            !events.is_empty(),
            "Events should be recorded from plain std::thread"
        );
        let has_span = events
            .iter()
            .any(|(t, b)| t == "observation-create" && b["name"] == "worker_thread_span");
        assert!(
            has_span,
            "observation-create for worker_thread_span should be recorded"
        );
        let has_close = events.iter().any(|(t, _)| t == "observation-update");
        assert!(
            has_close,
            "observation-update for span close should be recorded"
        );
    }
}
