// Minimal Server-Sent Events parser for fetch() streams.
//
// EventSource cannot send a POST body or an Authorization header, so the
// streaming endpoint is read with fetch() and parsed here.

export interface RawEvent {
  event: string;
  data: string;
}

/** Incremental parser: feed it decoded text chunks, get complete events back. */
export class SseParser {
  private buffer = "";
  private event = "message";
  private data: string[] = [];

  push(chunk: string): RawEvent[] {
    this.buffer += chunk;
    const out: RawEvent[] = [];
    // Lines can end in \n, \r\n or \r.
    const lines = this.buffer.split(/\r\n|\n|\r/);
    this.buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line === "") {
        if (this.data.length) out.push({ event: this.event, data: this.data.join("\n") });
        this.event = "message";
        this.data = [];
        continue;
      }
      if (line.startsWith(":")) continue; // comment / keep-alive ping
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") this.event = value || "message";
      else if (field === "data") this.data.push(value);
    }
    return out;
  }
}

export async function* readSse(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<RawEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    while (true) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      for (const event of parser.push(decoder.decode(value, { stream: true }))) yield event;
    }
    for (const event of parser.push(decoder.decode() + "\n\n")) yield event;
  } finally {
    reader.releaseLock();
  }
}
