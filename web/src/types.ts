// Shapes returned by the FastAPI backend (see app/models/schemas.py).

export type Mode = "agent" | "tutor";
export type AnswerStatus = "pending" | "completed" | "verified" | "unverified" | "failed";

export interface Session {
  access_token: string;
  user_id: string;
  username: string;
}

export interface Step {
  sequence: number;
  node: string;
  status: "ok" | "error";
  routing_decision?: string | null;
  latency_ms: number;
  output?: Record<string, unknown>;
  error?: string | null;
  /** Present on live SSE step events. */
  label?: string | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  answer_status?: AnswerStatus | null;
  fallback_reason?: string | null;
  confidence?: number | null;
  retry_count?: number | null;
  latency_ms?: number | null;
  error?: string | null;
  trace: Step[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  subject?: string | null;
  status: string;
  mode: Mode;
  agent_model?: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
}

export interface Page<T> {
  items: T[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  content_type?: string | null;
  size_bytes: number;
  char_count: number;
  chunk_count: number;
  created_at: string;
}

export interface AgentConfig {
  name: string;
  backend: string;
  auth: string;
  default_model: string;
  models: string[];
  effort: string;
  thinking: string;
  tools: string[];
  mcp_servers: string[];
  max_budget_usd: number;
}

export interface Health {
  status: string;
  database: string;
  llm_provider: string;
  llm_model: string;
  agent_backend?: string | null;
  agent_model?: string | null;
  agent_auth?: string | null;
  version: string;
}

export interface DoneEvent {
  conversation_id: string;
  assistant_message_id: string;
  answer_status: AnswerStatus;
  fallback_reason?: string | null;
  confidence?: number | null;
  retry_count?: number;
  latency_ms?: number | null;
  content: string;
}

export type StreamEvent =
  | { name: "step"; data: Step }
  | { name: "token"; data: { text: string } }
  | { name: "done"; data: DoneEvent }
  | { name: "error"; data: { code: string; message: string } };
