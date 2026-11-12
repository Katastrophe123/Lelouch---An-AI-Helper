import { useState } from "react";
import { modelName, money, seconds } from "../format";
import type { Message, Step } from "../types";
import { ActivityRail } from "./ActivityRail";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";

const STATUS: Record<string, { label: string; tone: "gold" | "emerald" | "warn" | "error" }> = {
  completed: { label: "Agent answer", tone: "gold" },
  verified: { label: "Self-checked", tone: "emerald" },
  unverified: { label: "Not verified, check it yourself", tone: "warn" },
  failed: { label: "Didn't finish", tone: "error" },
};

function costOf(trace: Step[]): number | undefined {
  const last = trace[trace.length - 1];
  const cost = last?.output?.cost_usd;
  return typeof cost === "number" ? cost : undefined;
}

export function UserMessage({ content }: { content: string }) {
  return (
    <article className="msg msg-user">
      <div className="msg-user-bubble">{content}</div>
    </article>
  );
}

export function AssistantMessage({ message, model }: { message: Message; model?: string | null }) {
  const [copied, setCopied] = useState(false);
  const status = message.answer_status ? STATUS[message.answer_status] : undefined;
  const cost = costOf(message.trace);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  };

  return (
    <article className="msg msg-assistant">
      <ActivityRail steps={message.trace} />
      <Markdown text={message.content} />
      <footer className="msg-meta">
        {status && (
          <span className="badge" data-tone={status.tone}>
            {status.label}
          </span>
        )}
        {typeof message.confidence === "number" && (
          <span className="meta-item">{Math.round(message.confidence * 100)}% confident</span>
        )}
        {model && message.answer_status === "completed" && <span className="meta-item">{modelName(model)}</span>}
        {typeof message.latency_ms === "number" && <span className="meta-item">{seconds(message.latency_ms)}</span>}
        {cost !== undefined && cost > 0 && <span className="meta-item">{money(cost)} of credit</span>}
        <button type="button" className="meta-action" onClick={copy}>
          <Icon name={copied ? "check" : "copy"} size={14} />
          {copied ? "Copied" : "Copy"}
        </button>
      </footer>
    </article>
  );
}

export interface LiveTurn {
  question: string;
  steps: Step[];
  text: string;
  error?: string;
}

export function LiveAssistant({ turn }: { turn: LiveTurn }) {
  return (
    <article className="msg msg-assistant" aria-live="polite" aria-busy={!turn.error}>
      <ActivityRail steps={turn.steps} live={!turn.error} />
      {turn.text ? <Markdown text={turn.text} /> : !turn.error && <p className="thinking-line">Working on it</p>}
      {turn.error && (
        <p className="msg-error" role="alert">
          <Icon name="alert" size={16} /> {turn.error}
        </p>
      )}
    </article>
  );
}
