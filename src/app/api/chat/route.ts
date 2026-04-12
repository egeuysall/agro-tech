import { NextResponse } from "next/server";

const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";
const DEFAULT_MODEL = "gpt-5.4-nano";
const MAX_MESSAGE_CHARS = 4000;
const MAX_HISTORY_MESSAGES = 20;
const MAX_ATTACHMENTS = 6;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const FORBIDDEN_TEXT_REGEX = /[()"'\u201c\u201d\u2018\u2019]/g;
const EM_DASH_REGEX = /\u2014/g;
const EMOJI_REGEX = /[\p{Extended_Pictographic}\p{Emoji_Presentation}]/gu;

type IncomingMessage = {
  role?: unknown;
  content?: unknown;
};

type IncomingAttachment = {
  name?: unknown;
  type?: unknown;
  size?: unknown;
  dataUrl?: unknown;
};

function normalizeString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeDelta(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function sanitizeAssistantText(text: string): string {
  return text
    .replace(FORBIDDEN_TEXT_REGEX, "")
    .replace(EM_DASH_REGEX, " ")
    .replace(EMOJI_REGEX, "")
    .replace(/\uFE0F/gu, "");
}

function isValidImageDataUrl(value: string): boolean {
  return /^data:image\/[a-zA-Z0-9.+-]+;base64,[a-zA-Z0-9+/=]+$/.test(value);
}

function parseBase64ByteLength(dataUrl: string): number {
  const base64 = dataUrl.split(",", 2)[1] ?? "";
  const padding = base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0;
  return Math.floor((base64.length * 3) / 4) - padding;
}

function normalizeMessages(raw: unknown): Array<{ role: string; content: string }> {
  if (!Array.isArray(raw)) return [];

  return raw
    .slice(-MAX_HISTORY_MESSAGES)
    .map((entry) => {
      const message = entry as IncomingMessage;
      const role = normalizeString(message.role);
      const content = normalizeString(message.content);
      if (!content) return null;

      if (role === "assistant") return { role: "assistant", content };
      if (role === "system") return { role: "system", content };
      return { role: "user", content };
    })
    .filter((entry): entry is { role: string; content: string } => Boolean(entry));
}

function normalizeAttachments(raw: unknown): Array<{
  name: string;
  type: string;
  size: number;
  dataUrl: string;
}> {
  if (!Array.isArray(raw)) return [];

  return raw
    .slice(0, MAX_ATTACHMENTS)
    .map((entry) => {
      const attachment = entry as IncomingAttachment;
      const name = normalizeString(attachment.name).slice(0, 120) || "image";
      const type = normalizeString(attachment.type);
      const dataUrl = normalizeString(attachment.dataUrl);
      const size =
        typeof attachment.size === "number" && Number.isFinite(attachment.size)
          ? attachment.size
          : parseBase64ByteLength(dataUrl);

      if (!type.startsWith("image/")) return null;
      if (!isValidImageDataUrl(dataUrl)) return null;
      if (size <= 0 || size > MAX_IMAGE_BYTES) return null;

      return { name, type, size, dataUrl };
    })
    .filter(
      (
        entry,
      ): entry is {
        name: string;
        type: string;
        size: number;
        dataUrl: string;
      } => Boolean(entry),
    );
}

function extractResponseText(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const record = payload as Record<string, unknown>;

  const outputText = normalizeString(record.output_text);
  if (outputText) return outputText;

  const output = record.output;
  if (!Array.isArray(output)) return "";

  const chunks: string[] = [];
  for (const item of output) {
    if (!item || typeof item !== "object") continue;
    const itemRecord = item as Record<string, unknown>;
    const content = itemRecord.content;
    if (!Array.isArray(content)) continue;

    for (const piece of content) {
      if (!piece || typeof piece !== "object") continue;
      const pieceRecord = piece as Record<string, unknown>;
      const text = normalizeString(pieceRecord.text);
      if (text) chunks.push(text);
    }
  }

  return chunks.join("\n").trim();
}

function extractStreamError(eventPayload: Record<string, unknown>): string {
  const nestedError = eventPayload.error;
  if (nestedError && typeof nestedError === "object") {
    const errorRecord = nestedError as Record<string, unknown>;
    const message = normalizeString(errorRecord.message);
    if (message) return message;
  }

  const message = normalizeString(eventPayload.message);
  if (message) return message;

  return "Streaming request failed.";
}

function createOpenAIPayload(args: {
  message: string;
  history: Array<{ role: string; content: string }>;
  attachments: Array<{ name: string; type: string; size: number; dataUrl: string }>;
  model: string;
  stream: boolean;
}) {
  const transcript = args.history
    .map((entry) => `${entry.role.toUpperCase()}: ${entry.content}`)
    .join("\n\n");

  const textBlock = transcript
    ? `Conversation history:\n${transcript}\n\nCurrent user message:\n${args.message}`
    : args.message;

  const userContent: Array<Record<string, string>> = [
    {
      type: "input_text",
      text: textBlock,
    },
    ...args.attachments.map((image) => ({
      type: "input_image",
      image_url: image.dataUrl,
    })),
  ];

  return {
    model: args.model,
    stream: args.stream,
    input: [
      {
        role: "user",
        content: userContent,
      },
    ],
    // Keep reasoning lightweight for faster first-token latency.
    reasoning: {
      effort: "none",
    },
    instructions:
      "You are AgroLens Assistant. Write clear practical farming advice in a human tone. Do not use parentheses. Do not use em dashes. Do not use quote marks. Do not use emojis. Keep responses concise and actionable.",
  };
}

function sseEncode(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

async function streamOpenAIResponse(
  source: ReadableStream<Uint8Array>,
  model: string,
): Promise<ReadableStream<Uint8Array>> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = source.getReader();
      let buffer = "";
      let finished = false;

      const send = (payload: unknown) => {
        controller.enqueue(encoder.encode(sseEncode(payload)));
      };

      send({ type: "meta", model });

      try {
        while (!finished) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          let boundaryIndex = buffer.indexOf("\n\n");
          while (boundaryIndex !== -1) {
            const eventBlock = buffer.slice(0, boundaryIndex);
            buffer = buffer.slice(boundaryIndex + 2);

            const lines = eventBlock
              .split("\n")
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trim());

            for (const line of lines) {
              if (!line) continue;
              if (line === "[DONE]") {
                send({ type: "done", model });
                finished = true;
                break;
              }

              let eventPayload: Record<string, unknown>;
              try {
                eventPayload = JSON.parse(line) as Record<string, unknown>;
              } catch {
                continue;
              }

              const eventType = normalizeString(eventPayload.type);

              if (eventType === "response.output_text.delta") {
                const delta = normalizeDelta(eventPayload.delta);
                const safeDelta = sanitizeAssistantText(delta);
                if (safeDelta.length) {
                  send({ type: "delta", text: safeDelta });
                }
              } else if (
                eventType === "error" ||
                eventType === "response.failed"
              ) {
                send({ type: "error", error: extractStreamError(eventPayload) });
                finished = true;
                break;
              } else if (eventType === "response.completed") {
                const responseRecord = eventPayload.response as
                  | Record<string, unknown>
                  | undefined;
                const completedModel = normalizeString(responseRecord?.model) || model;
                send({ type: "done", model: completedModel });
                finished = true;
                break;
              }
            }

            boundaryIndex = buffer.indexOf("\n\n");
          }
        }

        if (!finished) {
          send({ type: "done", model });
        }
      } catch (caughtError) {
        const message =
          caughtError instanceof Error ? caughtError.message : "Streaming failed.";
        send({ type: "error", error: message });
      } finally {
        reader.releaseLock();
        controller.close();
      }
    },
  });
}

export async function POST(request: Request) {
  const apiKey = process.env.OPENAI_API_KEY;
  const model = normalizeString(process.env.OPENAI_MODEL) || DEFAULT_MODEL;

  if (!apiKey) {
    return NextResponse.json(
      { error: "Missing OPENAI_API_KEY environment variable." },
      { status: 500 },
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const body = (payload ?? {}) as Record<string, unknown>;
  const message = normalizeString(body.message);
  const history = normalizeMessages(body.messages);
  const attachments = normalizeAttachments(body.attachments);
  const wantsStream = body.stream === true;

  if (!message) {
    return NextResponse.json(
      { error: "Message is required." },
      { status: 400 },
    );
  }

  if (message.length > MAX_MESSAGE_CHARS) {
    return NextResponse.json(
      { error: `Message exceeds ${MAX_MESSAGE_CHARS} characters.` },
      { status: 400 },
    );
  }

  const openaiPayload = createOpenAIPayload({
    message,
    history,
    attachments,
    model,
    stream: wantsStream,
  });

  const response = await fetch(OPENAI_RESPONSES_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(openaiPayload),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    return NextResponse.json(
      {
        error: `OpenAI request failed (${response.status}).`,
        details: errorBody.slice(0, 1200),
      },
      { status: 502 },
    );
  }

  if (wantsStream) {
    if (!response.body) {
      return NextResponse.json(
        { error: "OpenAI did not return a stream body." },
        { status: 502 },
      );
    }

    const streamed = await streamOpenAIResponse(response.body, model);
    return new Response(streamed, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  }

  const data: unknown = await response.json();
  const assistantMessage = sanitizeAssistantText(extractResponseText(data));

  if (!assistantMessage) {
    return NextResponse.json(
      { error: "OpenAI returned an empty response." },
      { status: 502 },
    );
  }

  return NextResponse.json({
    message: assistantMessage,
    model,
  });
}
