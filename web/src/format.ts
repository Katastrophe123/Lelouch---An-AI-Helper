// Plain-language descriptions of what the agent (or tutor) did at each step.

import type { Step } from "./types";

export interface StepView {
  title: string;
  detail?: string;
  kind: "start" | "think" | "web" | "browser" | "docs" | "plan" | "delegate" | "tool" | "answer" | "tutor";
}

const str = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined;

const BROWSER_VERBS: Record<string, string> = {
  browser_navigate: "Opened a page",
  browser_navigate_back: "Went back",
  browser_click: "Clicked",
  browser_type: "Typed",
  browser_fill_form: "Filled in a form",
  browser_select_option: "Chose an option",
  browser_press_key: "Pressed a key",
  browser_hover: "Hovered",
  browser_snapshot: "Read the page",
  browser_take_screenshot: "Took a screenshot",
  browser_wait_for: "Waited for the page",
  browser_tabs: "Switched tabs",
  browser_close: "Closed the browser",
  browser_evaluate: "Ran a script on the page",
};

const TUTOR: Record<string, string> = {
  understand: "Understood the question",
  solve: "Worked out a solution",
  self_check: "Checked the answer",
  explain: "Wrote the explanation",
  fallback: "Prepared a partial answer",
};

export function describeStep(step: Step): StepView {
  const out = step.output ?? {};
  const input = str(out.input);
  const tool = str(out.tool) ?? "";

  switch (step.node) {
    case "agent_start": {
      const servers = out.mcp_servers as Record<string, string> | undefined;
      const names = servers ? Object.keys(servers).filter((n) => n !== "knowledge") : [];
      return {
        kind: "start",
        title: `Started on ${modelName(str(out.model)) || "Claude"}`,
        detail: names.length ? `Connected: ${names.join(", ")}` : undefined,
      };
    }
    case "thinking":
      return { kind: "think", title: "Thought it through", detail: str(out.preview) };
    case "answer": {
      if (step.status === "error") return { kind: "answer", title: "Stopped", detail: step.error ?? undefined };
      const cost = typeof out.cost_usd === "number" && out.cost_usd > 0 ? `, ${money(out.cost_usd)}` : "";
      const ms = typeof out.total_ms === "number" ? out.total_ms : undefined;
      return { kind: "answer", title: "Wrote the answer", detail: ms !== undefined ? `${seconds(ms)}${cost}` : undefined };
    }
    case "tool":
      break;
    default:
      if (TUTOR[step.node]) {
        let detail: string | undefined;
        if (step.node === "self_check") {
          const passed = out.passed === true;
          const confidence = typeof out.confidence === "number" ? ` (${Math.round(out.confidence * 100)}% sure)` : "";
          detail = passed ? `Passed${confidence}` : `Rejected${confidence}`;
        }
        return { kind: "tutor", title: TUTOR[step.node] ?? step.node, detail };
      }
      return { kind: "tool", title: str(step.label) ?? str(out.label) ?? step.node };
  }

  if (tool === "WebSearch") return { kind: "web", title: "Searched the web", detail: input };
  if (tool === "WebFetch") return { kind: "web", title: "Read a page", detail: input };
  if (tool === "Task" || tool === "Agent") return { kind: "delegate", title: "Asked the researcher", detail: input };
  if (/^(TaskCreate|TaskUpdate|TaskList|TaskGet|TodoWrite)$/.test(tool)) return { kind: "plan", title: "Updated the plan", detail: undefined };
  if (tool.endsWith("search_my_documents")) return { kind: "docs", title: "Searched your documents", detail: input };
  const match = /^mcp__([^_]+(?:_[^_]+)*)__(.+)$/.exec(tool);
  if (match) {
    const [, server, name] = match as unknown as [string, string, string];
    if (server === "playwright") return { kind: "browser", title: BROWSER_VERBS[name] ?? `Used the browser (${name})`, detail: input };
    return { kind: "tool", title: `Used ${server}`, detail: input ? `${name}: ${input}` : name };
  }
  return { kind: "tool", title: str(out.label) ?? (tool || "Used a tool"), detail: input };
}

export function seconds(ms: number): string {
  if (ms < 1000) return "under a second";
  const s = ms / 1000;
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
}

export function money(usd: number): string {
  return usd < 0.01 ? "under 1¢" : `$${usd.toFixed(2)}`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function relativeDay(iso: string, now = new Date()): string {
  const date = new Date(iso);
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOf(now) - startOf(date)) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "This week";
  if (days < 31) return "This month";
  return "Older";
}

export function modelName(model?: string | null): string {
  if (!model) return "";
  const known: Record<string, string> = {
    "claude-opus-5-5": "Opus 5.5",
    "claude-fable-5-2": "Fable 5.2",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-haiku-5-5": "Haiku 5.5",
  };
  return known[model] ?? model.replace(/^claude-/, "");
}
