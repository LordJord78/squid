// Server-Sent Events (SSE): the server keeps the reply open and writes "data: <text>" plus a blank
// line whenever it has something new. Both LM Studio and llama.cpp stream chat replies this way.

// Reads the data of each event as soon as it arrives. Network chunks don't line up with events
// (one chunk can hold three events, or half of one), so text is buffered until a blank line ends an event.
export async function* sseData(body: AsyncIterable<Uint8Array | string>): AsyncGenerator<string> {
  // stream: true keeps a character whose bytes were split between two chunks (an emoji is 4 bytes)
  // until the rest arrives, instead of turning each half into "�".
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of body) {
    // Windows-style "\r\n" line ends count as "\n". Replaced across the whole buffer, so a "\r" at the
    // end of one chunk still pairs with the "\n" that starts the next.
    buffer = (buffer + (typeof chunk === "string" ? chunk : decoder.decode(chunk, { stream: true }))).replace(/\r\n/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = event.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).replace(/^ /, ""));
      if (data.length) yield data.join("\n");
    }
  }
}

// One event, written the way LM Studio writes them.
export const sseEvent = (data: string) => `data: ${data}\n\n`;
