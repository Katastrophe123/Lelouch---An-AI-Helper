// Typed client for the FastAPI backend. Same origin in production (FastAPI
// serves this app), proxied by Vite in development.

import { readSse } from "./sse";
import type {
  AgentConfig,
  Conversation,
  ConversationSummary,
  DocumentInfo,
  Health,
  Mode,
  Page,
  Session,
  StreamEvent,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function errorFrom(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `The server answered ${response.status}.`;
  try {
    const body = await response.json();
    code = body?.error?.code ?? code;
    message = body?.error?.message ?? message;
  } catch {
    /* not JSON */
  }
  return new ApiError(response.status, code, message);
}

export class Api {
  constructor(private token: string | null = null) {}

  withToken(token: string | null): Api {
    return new Api(token);
  }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    return this.token ? { Authorization: `Bearer ${this.token}`, ...extra } : extra;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: this.headers((init.headers as Record<string, string>) ?? {}),
    });
    if (!response.ok) throw await errorFrom(response);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  private json<T>(path: string, method: string, body: unknown): Promise<T> {
    return this.request<T>(path, {
      method,
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
    });
  }

  health(): Promise<Health> {
    return this.request("/health");
  }

  signIn(username: string, password: string): Promise<Session> {
    return this.json("/auth/token", "POST", { username, password });
  }

  agentConfig(): Promise<AgentConfig> {
    return this.request("/agent/config");
  }

  listConversations(): Promise<Page<ConversationSummary>> {
    return this.request("/conversations?page=1&page_size=100");
  }

  getConversation(id: string): Promise<Conversation> {
    return this.request(`/conversations/${id}`);
  }

  createConversation(mode: Mode, model?: string): Promise<Conversation> {
    return this.json("/conversations", "POST", mode === "agent" ? { mode, model } : { mode });
  }

  listDocuments(): Promise<DocumentInfo[]> {
    return this.request("/documents");
  }

  uploadDocument(file: File): Promise<DocumentInfo> {
    const form = new FormData();
    form.append("file", file, file.name);
    return this.request("/documents", { method: "POST", body: form });
  }

  deleteDocument(id: string): Promise<void> {
    return this.request(`/documents/${id}`, { method: "DELETE" });
  }

  /** Ask a question and yield step / token / done / error events as they arrive. */
  async *ask(conversationId: string, content: string, signal: AbortSignal): AsyncGenerator<StreamEvent> {
    const response = await fetch(`${BASE}/conversations/${conversationId}/messages/stream`, {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify({ content }),
      signal,
    });
    if (!response.ok || !response.body) throw await errorFrom(response);
    for await (const raw of readSse(response.body, signal)) {
      if (!["step", "token", "done", "error"].includes(raw.event)) continue;
      try {
        yield { name: raw.event, data: JSON.parse(raw.data) } as StreamEvent;
      } catch {
        /* skip malformed frame */
      }
    }
  }
}

const STORAGE_KEY = "lelouch.session";

export function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session | null): void {
  try {
    if (session) localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable: the session lives for this tab only */
  }
}
