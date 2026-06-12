use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParam, CallToolResult, Content, JsonObject, ListToolsResult, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::io::stdio;
use rmcp::{serve_server, RoleServer};
use std::sync::Arc;

/// Servidor MCP com ferramentas matemáticas — Guilda de IA S07
///
/// Roda via stdio (stdin/stdout), igual ao servidor Python.
/// Para usar: rode o binário e conecte via MCP stdio transport.

#[derive(Debug, Clone)]
struct MathServer;

impl ServerHandler for MathServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::default(),
            capabilities: rmcp::model::ServerCapabilities::default(),
            server_info: rmcp::model::Implementation {
                name: "math-server-mcp".into(),
                version: "0.1.0".into(),
            },
            instructions: None,
        }
    }

    fn list_tools(
        &self,
        _request: rmcp::model::PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<
        Output = Result<ListToolsResult, rmcp::Error>,
    > + Send + '_ {
        let add_schema: Arc<JsonObject> = {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), serde_json::Value::String("object".into()));
            let mut props = serde_json::Map::new();
            props.insert("a".into(), serde_json::json!({"type": "integer", "description": "First number"}));
            props.insert("b".into(), serde_json::json!({"type": "integer", "description": "Second number"}));
            obj.insert("properties".into(), serde_json::Value::Object(props));
            obj.insert("required".into(), serde_json::json!(["a", "b"]));
            Arc::new(obj)
        };
        let mul_schema: Arc<JsonObject> = add_schema.clone();
        std::future::ready(Ok(ListToolsResult {
            tools: vec![
                Tool::new("add", "Add two numbers together", add_schema),
                Tool::new("multiply", "Multiply two numbers together", mul_schema),
            ],
            ..Default::default()
        }))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, rmcp::Error>> + Send + '_ {
        let name = request.name.to_string();
        let args = request.arguments;
        async move {
            let result = match name.as_str() {
                "add" => {
                    let a: i64 = args.as_ref()
                        .and_then(|v| v.get("a"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let b: i64 = args.as_ref()
                        .and_then(|v| v.get("b"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    format!("{}", a + b)
                }
                "multiply" => {
                    let a: i64 = args.as_ref()
                        .and_then(|v| v.get("a"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let b: i64 = args.as_ref()
                        .and_then(|v| v.get("b"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    format!("{}", a * b)
                }
                _ => format!("Unknown tool: {}", name),
            };
            Ok(CallToolResult::success(vec![Content::text(result)]))
        }
    }
}

#[tokio::main]
async fn main() {
    let (stdin, stdout) = stdio();
    let service = serve_server(MathServer, (stdin, stdout))
        .await
        .expect("Failed to start MCP server");
    service.waiting().await.expect("Server error");
}