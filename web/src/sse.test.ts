import { describe, expect, it } from "vitest";
import { readSse, SseParser } from "./sse";

describe("SseParser", () => {
  it("parses named events with JSON data", () => {
    const parser = new SseParser();
    const events = parser.push('event: step\ndata: {"node":"tool"}\n\nevent: token\ndata: {"text":"hi"}\n\n');
    expect(events).toEqual([
      { event: "step", data: '{"node":"tool"}' },
      { event: "token", data: '{"text":"hi"}' },
    ]);
  });

  it("handles events split across chunks and CRLF line endings", () => {
    const parser = new SseParser();
    expect(parser.push("event: to")).toEqual([]);
    expect(parser.push('ken\r\ndata: {"text":')).toEqual([]);
    expect(parser.push('"a"}\r\n\r\n')).toEqual([{ event: "token", data: '{"text":"a"}' }]);
  });

  it("ignores keep-alive comments and joins multi-line data", () => {
    const parser = new SseParser();
    const events = parser.push(": ping\n\ndata: line one\ndata: line two\n\n");
    expect(events).toEqual([{ event: "message", data: "line one\nline two" }]);
  });

  it("resets the event name after each dispatch", () => {
    const parser = new SseParser();
    const events = parser.push("event: done\ndata: 1\n\ndata: 2\n\n");
    expect(events.map((e) => e.event)).toEqual(["done", "message"]);
  });
});

describe("readSse", () => {
  it("reads a byte stream to the end, flushing a trailing event", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode("event: step\ndata: a\n\n"));
        controller.enqueue(encoder.encode("event: done\ndata: b"));
        controller.close();
      },
    });
    const seen = [];
    for await (const event of readSse(body)) seen.push(event);
    expect(seen).toEqual([
      { event: "step", data: "a" },
      { event: "done", data: "b" },
    ]);
  });
});
