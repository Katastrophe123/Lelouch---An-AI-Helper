import { describe, expect, it } from "vitest";
import { bytes, describeStep, modelName, money, relativeDay, seconds } from "./format";
import type { Step } from "./types";

const step = (node: string, output: Record<string, unknown> = {}, extra: Partial<Step> = {}): Step => ({
  sequence: 1,
  node,
  status: "ok",
  latency_ms: 0,
  output,
  ...extra,
});

describe("describeStep", () => {
  it("names web, document, browser and delegation tools in plain words", () => {
    expect(describeStep(step("tool", { tool: "WebSearch", input: "plassey" }))).toMatchObject({
      kind: "web",
      title: "Searched the web",
      detail: "plassey",
    });
    expect(describeStep(step("tool", { tool: "mcp__knowledge__search_my_documents", input: "q" })).title).toBe(
      "Searched your documents",
    );
    expect(describeStep(step("tool", { tool: "mcp__playwright__browser_navigate", input: "https://x" }))).toMatchObject({
      kind: "browser",
      title: "Opened a page",
    });
    expect(describeStep(step("tool", { tool: "Task", input: "dig into it" })).title).toBe("Asked the researcher");
    expect(describeStep(step("tool", { tool: "TaskCreate" })).kind).toBe("plan");
    expect(describeStep(step("tool", { tool: "mcp__github__create_issue", input: "bug" }))).toMatchObject({
      title: "Used github",
      detail: "create_issue: bug",
    });
  });

  it("summarises start, answer and tutor steps", () => {
    expect(describeStep(step("agent_start", { model: "claude-opus-5-5", mcp_servers: { playwright: "connected", knowledge: "connected" } }))).toMatchObject({
      title: "Started on Opus 5.5",
      detail: "Connected: playwright",
    });
    expect(describeStep(step("answer", { total_ms: 4200, cost_usd: 0.12 })).detail).toBe("4.2 s, $0.12");
    expect(describeStep(step("answer", {}, { status: "error", error: "budget" })).title).toBe("Stopped");
    expect(describeStep(step("self_check", { passed: true, confidence: 0.88 })).detail).toBe("Passed (88% sure)");
  });
});

describe("formatters", () => {
  it("formats durations, money, sizes and model names", () => {
    expect(seconds(300)).toBe("under a second");
    expect(seconds(42_000)).toBe("42 s");
    expect(seconds(125_000)).toBe("2 min 5 s");
    expect(money(0.004)).toBe("under 1¢");
    expect(bytes(2048)).toBe("2 KB");
    expect(modelName("claude-fable-5-2")).toBe("Fable 5.2");
    expect(modelName("claude-x-1")).toBe("x-1");
  });

  it("groups dates", () => {
    const now = new Date("2026-11-12T12:00:00");
    expect(relativeDay("2026-11-12T08:00:00", now)).toBe("Today");
    expect(relativeDay("2026-11-11T08:00:00", now)).toBe("Yesterday");
    expect(relativeDay("2026-09-01T08:00:00", now)).toBe("Older");
  });
});
