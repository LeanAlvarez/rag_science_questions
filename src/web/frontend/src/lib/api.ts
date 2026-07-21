import type { StreamEvent } from "./types";

/**
 * POST /api/ask/stream and yield SSE events as they arrive.
 *
 * We can't use the native `EventSource` API because it's GET-only, and we
 * need to POST a JSON body. So we do it by hand: `fetch` for the request,
 * a `ReadableStream` reader on the response body, and a small parser that
 * splits on the SSE frame delimiter (`\n\n`).
 *
 * The generator is cancellable — call `.return()` on it, or use it in a
 * `for await` inside an AbortController-aware caller.
 */
export async function* streamAsk(
  question: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent, void, void> {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(
      `Stream request failed: HTTP ${response.status} ${response.statusText}`,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Split off any complete SSE frames (separated by blank line) and
      // keep the trailing partial frame in the buffer for the next read.
      let sep = buffer.indexOf("\n\n");
      while (sep !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const parsed = parseSseFrame(raw);
        if (parsed) yield parsed;
        sep = buffer.indexOf("\n\n");
      }
    }
  } finally {
    // Best-effort teardown — makes cancellation immediate instead of waiting
    // for the server to close its end.
    try {
      await reader.cancel();
    } catch {
      /* swallow — we're already unwinding */
    }
  }
}

/**
 * Parses one SSE frame ("event: X\ndata: Y") into a StreamEvent, or returns
 * null for a keepalive / malformed / comment-only frame.
 */
function parseSseFrame(raw: string): StreamEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of raw.split("\n")) {
    if (!line || line.startsWith(":")) continue; // comment / keepalive
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return null;

  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }

  // Trust the server's event tag. The `as StreamEvent` cast is safe because
  // our server only ever emits the four known event kinds, but callers
  // should still switch on `.event` before using `.data`.
  return { event, data } as StreamEvent;
}
