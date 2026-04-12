"use client";

import Image from "next/image";
import Link from "next/link";
import {
  DragEvent,
  FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Streamdown } from "streamdown";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";

type MessageRole = "user" | "assistant" | "system";

type ChatAttachment = {
  name: string;
  type: string;
  size: number;
  dataUrl: string;
};

type ChatMessage = {
  id: string;
  role: MessageRole;
  content: string;
  attachments?: ChatAttachment[];
};

type ImageAttachment = {
  id: string;
  name: string;
  type: string;
  size: number;
  dataUrl: string;
};

type ChatSession = {
  id: string;
  title: string;
  messages: ChatMessage[];
  activeModel: string;
  createdAt: number;
  updatedAt: number;
};

type PersistedMeta = {
  key: string;
  activeSessionId: string;
};

const API_URL_KEY = "agro-tech:chat-api-url";
const DEFAULT_API_URL = process.env.NEXT_PUBLIC_CHAT_API_URL ?? "";
const DEFAULT_FALLBACK_MODEL = "gpt-5.4-nano";
const MAX_MESSAGE_CHARS = 4000;
const MAX_ATTACHMENTS = 6;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

const CHAT_DB_NAME = "agrolens-chat-db";
const CHAT_DB_VERSION = 2;
const CHAT_SESSIONS_STORE = "chat_sessions";
const CHAT_META_STORE = "chat_meta";
const CHAT_META_KEY = "app";

const LEGACY_CHAT_STORE_NAME = "chat_state";
const LEGACY_CHAT_STATE_KEY = "main";

const FORBIDDEN_TEXT_REGEX = /[()"'\u201c\u201d\u2018\u2019]/g;
const EM_DASH_REGEX = /\u2014/g;
const EMOJI_REGEX = /[\p{Extended_Pictographic}\p{Emoji_Presentation}]/gu;

function sanitizeAssistantText(text: string): string {
  return text
    .replace(FORBIDDEN_TEXT_REGEX, "")
    .replace(EM_DASH_REGEX, " ")
    .replace(EMOJI_REGEX, "")
    .replace(/\uFE0F/gu, "");
}

function createMessageId(prefix: MessageRole | "attachment" | "session"): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isValidApiUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function normalizeModel(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeDelta(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function formatRelativeTime(timestamp: number): string {
  const delta = Date.now() - timestamp;
  if (delta < 60_000) return "now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h`;
  return `${Math.floor(delta / 86_400_000)}d`;
}

function deriveSessionTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find(
    (message) => message.role === "user" && message.content.trim().length > 0,
  );

  if (!firstUser) return "New chat";

  const normalized = firstUser.content.replace(/\s+/g, " ").trim();
  if (!normalized) return "New chat";

  return normalized.length > 48
    ? `${normalized.slice(0, 48).trimEnd()}...`
    : normalized;
}

function createInitialMessages(): ChatMessage[] {
  return [
    {
      id: "welcome-message",
      role: "assistant",
      content:
        "AgroLens is online. Leave API URL empty for built in OpenAI fallback, or set your own endpoint.",
    },
  ];
}

function createSession(seed?: Partial<ChatSession>): ChatSession {
  const now = Date.now();
  const messages =
    seed?.messages && seed.messages.length > 0
      ? seed.messages
      : createInitialMessages();

  return {
    id: seed?.id ?? createMessageId("session"),
    title: seed?.title ?? deriveSessionTitle(messages),
    messages,
    activeModel: seed?.activeModel ?? DEFAULT_FALLBACK_MODEL,
    createdAt: seed?.createdAt ?? now,
    updatedAt: seed?.updatedAt ?? now,
  };
}

function sortSessions(sessions: ChatSession[]): ChatSession[] {
  return [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
}

function updateSessionById(
  sessions: ChatSession[],
  sessionId: string,
  updater: (session: ChatSession) => ChatSession,
): ChatSession[] {
  return sortSessions(
    sessions.map((session) =>
      session.id === sessionId ? updater(session) : session,
    ),
  );
}

function normalizeAttachment(raw: unknown): ChatAttachment | null {
  if (!raw || typeof raw !== "object") return null;

  const record = raw as Record<string, unknown>;
  const name = typeof record.name === "string" ? record.name : "image";
  const type = typeof record.type === "string" ? record.type : "image/jpeg";
  const dataUrl = typeof record.dataUrl === "string" ? record.dataUrl : "";
  const size =
    typeof record.size === "number" && Number.isFinite(record.size)
      ? record.size
      : 0;

  if (!dataUrl.startsWith("data:image/")) return null;
  if (!type.startsWith("image/")) return null;

  return {
    name,
    type,
    size,
    dataUrl,
  };
}

function normalizeMessage(raw: unknown): ChatMessage | null {
  if (!raw || typeof raw !== "object") return null;

  const record = raw as Record<string, unknown>;
  const role = record.role;
  const content = record.content;

  if (role !== "assistant" && role !== "user" && role !== "system") {
    return null;
  }

  if (typeof content !== "string") return null;
  const safeContent =
    role === "assistant" ? sanitizeAssistantText(content) : content;

  const attachments = Array.isArray(record.attachments)
    ? record.attachments
        .map((attachment) => normalizeAttachment(attachment))
        .filter((attachment): attachment is ChatAttachment => Boolean(attachment))
    : [];

  return {
    id:
      typeof record.id === "string" && record.id.length > 0
        ? record.id
        : createMessageId(role),
    role,
    content: safeContent,
    attachments: attachments.length > 0 ? attachments : undefined,
  };
}

function normalizeSession(raw: unknown): ChatSession | null {
  if (!raw || typeof raw !== "object") return null;

  const record = raw as Record<string, unknown>;
  const id = typeof record.id === "string" ? record.id : "";
  if (!id) return null;

  const messages = Array.isArray(record.messages)
    ? record.messages
        .map((message) => normalizeMessage(message))
        .filter((message): message is ChatMessage => Boolean(message))
    : [];

  const safeMessages =
    messages.length > 0 ? messages : createSession({ id }).messages;

  const createdAt =
    typeof record.createdAt === "number" && Number.isFinite(record.createdAt)
      ? record.createdAt
      : Date.now();

  const updatedAt =
    typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt)
      ? record.updatedAt
      : createdAt;

  const activeModel = normalizeModel(record.activeModel) || DEFAULT_FALLBACK_MODEL;

  return {
    id,
    title:
      typeof record.title === "string" && record.title.trim().length > 0
        ? record.title
        : deriveSessionTitle(safeMessages),
    messages: safeMessages,
    activeModel,
    createdAt,
    updatedAt,
  };
}

function extractAssistantReply(payload: unknown): string {
  if (typeof payload === "string") {
    return payload;
  }

  if (!payload || typeof payload !== "object") {
    return "";
  }

  if (Array.isArray(payload)) {
    return payload.map(extractAssistantReply).filter(Boolean).join("\n");
  }

  const record = payload as Record<string, unknown>;

  const directKeys = [
    "message",
    "response",
    "reply",
    "output",
    "content",
    "text",
    "answer",
  ];

  for (const key of directKeys) {
    const value = record[key];
    const extracted = extractAssistantReply(value);
    if (extracted) {
      return extracted;
    }
  }

  const choices = record.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const firstChoice = choices[0];
    if (firstChoice && typeof firstChoice === "object") {
      const choiceRecord = firstChoice as Record<string, unknown>;
      const extracted =
        extractAssistantReply(choiceRecord.message) ||
        extractAssistantReply(choiceRecord.delta) ||
        extractAssistantReply(choiceRecord.text);
      if (extracted) {
        return extracted;
      }
    }
  }

  return JSON.stringify(payload);
}

function extractModel(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const record = payload as Record<string, unknown>;
  return normalizeModel(record.model);
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

function openChatDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(CHAT_DB_NAME, CHAT_DB_VERSION);

    request.onupgradeneeded = (event) => {
      const database = request.result;
      const oldVersion = event.oldVersion;

      if (!database.objectStoreNames.contains(CHAT_SESSIONS_STORE)) {
        database.createObjectStore(CHAT_SESSIONS_STORE, { keyPath: "id" });
      }

      if (!database.objectStoreNames.contains(CHAT_META_STORE)) {
        database.createObjectStore(CHAT_META_STORE, { keyPath: "key" });
      }

      if (
        oldVersion < 2 &&
        database.objectStoreNames.contains(LEGACY_CHAT_STORE_NAME) &&
        request.transaction
      ) {
        const tx = request.transaction;
        const legacyStore = tx.objectStore(LEGACY_CHAT_STORE_NAME);
        const sessionsStore = tx.objectStore(CHAT_SESSIONS_STORE);
        const metaStore = tx.objectStore(CHAT_META_STORE);
        const legacyRequest = legacyStore.get(LEGACY_CHAT_STATE_KEY);

        legacyRequest.onsuccess = () => {
          const legacy = legacyRequest.result as
            | { messages?: unknown; activeModel?: unknown }
            | undefined;

          if (!legacy || !Array.isArray(legacy.messages)) return;

          const migratedMessages = legacy.messages
            .map((message) => normalizeMessage(message))
            .filter((message): message is ChatMessage => Boolean(message));

          const session = createSession({
            messages:
              migratedMessages.length > 0
                ? migratedMessages
                : createInitialMessages(),
            activeModel:
              normalizeModel(legacy.activeModel) || DEFAULT_FALLBACK_MODEL,
          });

          sessionsStore.put(session);
          metaStore.put({ key: CHAT_META_KEY, activeSessionId: session.id });
        };
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Failed to open IndexedDB."));
  });
}

async function loadPersistedChatState(): Promise<{
  sessions: ChatSession[];
  activeSessionId: string;
} | null> {
  if (typeof window === "undefined" || !("indexedDB" in window)) return null;

  const db = await openChatDatabase();

  try {
    const state = await new Promise<{
      sessions: ChatSession[];
      activeSessionId: string;
    }>((resolve, reject) => {
      const tx = db.transaction([CHAT_SESSIONS_STORE, CHAT_META_STORE], "readonly");
      const sessionsStore = tx.objectStore(CHAT_SESSIONS_STORE);
      const metaStore = tx.objectStore(CHAT_META_STORE);
      const sessionsRequest = sessionsStore.getAll();
      const metaRequest = metaStore.get(CHAT_META_KEY);

      tx.oncomplete = () => {
        const sessions = Array.isArray(sessionsRequest.result)
          ? sessionsRequest.result
              .map((session) => normalizeSession(session))
              .filter((session): session is ChatSession => Boolean(session))
          : [];

        const sorted = sortSessions(sessions);
        const meta = metaRequest.result as PersistedMeta | undefined;
        const activeSessionId =
          meta && typeof meta.activeSessionId === "string"
            ? meta.activeSessionId
            : "";

        resolve({
          sessions: sorted,
          activeSessionId,
        });
      };

      tx.onerror = () =>
        reject(tx.error ?? new Error("Failed reading chat state."));
    });

    if (state.sessions.length === 0) {
      const defaultSession = createSession();
      return {
        sessions: [defaultSession],
        activeSessionId: defaultSession.id,
      };
    }

    const hasActive = state.sessions.some(
      (session) => session.id === state.activeSessionId,
    );

    return {
      sessions: state.sessions,
      activeSessionId: hasActive
        ? state.activeSessionId
        : state.sessions[0]?.id ?? "",
    };
  } finally {
    db.close();
  }
}

async function persistChatState(state: {
  sessions: ChatSession[];
  activeSessionId: string;
}): Promise<void> {
  if (typeof window === "undefined" || !("indexedDB" in window)) return;

  const db = await openChatDatabase();

  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction([CHAT_SESSIONS_STORE, CHAT_META_STORE], "readwrite");
      const sessionsStore = tx.objectStore(CHAT_SESSIONS_STORE);
      const metaStore = tx.objectStore(CHAT_META_STORE);

      sessionsStore.clear();
      for (const session of state.sessions) {
        sessionsStore.put(session);
      }

      metaStore.put({
        key: CHAT_META_KEY,
        activeSessionId: state.activeSessionId,
      } satisfies PersistedMeta);

      tx.oncomplete = () => resolve();
      tx.onerror = () =>
        reject(tx.error ?? new Error("Failed saving chat state."));
    });
  } finally {
    db.close();
  }
}

async function readSseStream(
  stream: ReadableStream<Uint8Array>,
  handlers: {
    onDelta: (delta: string) => boolean;
    onModel: (model: string) => void;
    onError: (error: string) => void;
  },
): Promise<{ hasText: boolean }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let hasText = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let boundaryIndex = buffer.indexOf("\n\n");
      while (boundaryIndex !== -1) {
        const eventBlock = buffer.slice(0, boundaryIndex);
        buffer = buffer.slice(boundaryIndex + 2);

        const dataLines = eventBlock
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim());

        for (const dataLine of dataLines) {
          if (!dataLine || dataLine === "[DONE]") continue;

          let eventPayload: Record<string, unknown>;
          try {
            eventPayload = JSON.parse(dataLine) as Record<string, unknown>;
          } catch {
            continue;
          }

          const type = normalizeModel(eventPayload.type);

          if (type === "delta") {
            const delta = normalizeDelta(eventPayload.text);
            if (!delta.length) continue;
            if (handlers.onDelta(delta)) {
              hasText = true;
            }
          } else if (type === "meta" || type === "done") {
            const model = normalizeModel(eventPayload.model);
            if (model) handlers.onModel(model);
          } else if (type === "error") {
            const errorMessage =
              normalizeModel(eventPayload.error) || "Streaming failed.";
            handlers.onError(errorMessage);
          }
        }

        boundaryIndex = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }

  return { hasText };
}

function MobileSidebarToggle() {
  const { toggleSidebar } = useSidebar();

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      className="h-8 md:hidden"
      onClick={toggleSidebar}
    >
      Menu
    </Button>
  );
}

export default function Home() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [sessions, setSessions] = useState<ChatSession[]>(() => [createSession()]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<ImageAttachment[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasLoadedState, setHasLoadedState] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);

  const orderedSessions = useMemo(() => sortSessions(sessions), [sessions]);
  const activeSession = useMemo(
    () =>
      orderedSessions.find((session) => session.id === activeSessionId) ??
      orderedSessions[0],
    [orderedSessions, activeSessionId],
  );

  useEffect(() => {
    if (!activeSession && orderedSessions.length > 0) {
      setActiveSessionId(orderedSessions[0].id);
      return;
    }

    if (!activeSessionId && activeSession) {
      setActiveSessionId(activeSession.id);
    }
  }, [activeSession, activeSessionId, orderedSessions]);

  useEffect(() => {
    let isMounted = true;

    void loadPersistedChatState()
      .then((state) => {
        if (!isMounted || !state) return;

        setSessions(state.sessions);
        setActiveSessionId(state.activeSessionId);
      })
      .catch(() => {
        // Non fatal. The app still works without persisted state.
      })
      .finally(() => {
        if (isMounted) {
          setHasLoadedState(true);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem(API_URL_KEY);
    if (saved && !DEFAULT_API_URL) {
      setApiUrl(saved);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const normalized = apiUrl.trim();
    if (normalized) {
      window.localStorage.setItem(API_URL_KEY, normalized);
      return;
    }
    window.localStorage.removeItem(API_URL_KEY);
  }, [apiUrl]);

  useEffect(() => {
    messageInputRef.current?.focus();
  }, []);

  useEffect(() => {
    const container = listRef.current;
    if (!container) return;
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [activeSession?.messages, isSending]);

  useEffect(() => {
    if (!hasLoadedState || !activeSession) return;

    const timer = window.setTimeout(() => {
      void persistChatState({
        sessions: orderedSessions,
        activeSessionId: activeSession.id,
      }).catch(() => {
        // Non fatal. Ignore persistence errors.
      });
    }, 200);

    return () => {
      window.clearTimeout(timer);
    };
  }, [orderedSessions, activeSession, hasLoadedState]);

  const addFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;

    const files = Array.from(fileList);
    const availableSlots = Math.max(0, MAX_ATTACHMENTS - attachments.length);
    if (availableSlots === 0) {
      setError(`You can attach up to ${MAX_ATTACHMENTS} images per request.`);
      return;
    }

    const nextFiles = files.slice(0, availableSlots);

    try {
      const parsed = await Promise.all(
        nextFiles.map(async (file) => {
          if (!file.type.startsWith("image/")) {
            throw new Error(`${file.name} is not an image file.`);
          }
          if (file.size > MAX_IMAGE_BYTES) {
            throw new Error(
              `${file.name} exceeds ${formatFileSize(MAX_IMAGE_BYTES)}.`,
            );
          }

          const dataUrl = await readFileAsDataUrl(file);
          return {
            id: createMessageId("attachment"),
            name: file.name,
            type: file.type,
            size: file.size,
            dataUrl,
          } satisfies ImageAttachment;
        }),
      );

      setAttachments((current) => [...current, ...parsed]);
      setError(null);

      if (files.length > availableSlots) {
        setError(
          `Only first ${availableSlots} file(s) were added due to attachment limit.`,
        );
      }
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : "Failed to attach image files.";
      setError(message);
    }
  };

  const onDropFiles = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingFiles(false);
    await addFiles(event.dataTransfer.files);
  };

  const createNewChat = () => {
    const session = createSession();
    setSessions((current) => sortSessions([session, ...current]));
    setActiveSessionId(session.id);
    setInput("");
    setAttachments([]);
    setError(null);
  };

  const submitPrompt = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (isSending || !activeSession) return;

    const prompt = input.trim();
    const customApiUrl = apiUrl.trim();
    const targetUrl = customApiUrl || "/api/chat";
    const shouldStream = !customApiUrl;

    if (!prompt) return;

    if (prompt.length > MAX_MESSAGE_CHARS) {
      setError(`Message exceeds ${MAX_MESSAGE_CHARS} characters.`);
      return;
    }

    if (customApiUrl && !isValidApiUrl(customApiUrl)) {
      setError("Custom API URL must be a valid http or https URL.");
      return;
    }

    setError(null);

    const targetSessionId = activeSession.id;
    const outgoingAttachments = attachments.map(({ name, type, size, dataUrl }) => ({
      name,
      type,
      size,
      dataUrl,
    }));

    const nextUserMessage: ChatMessage = {
      id: createMessageId("user"),
      role: "user",
      content: prompt,
      attachments: outgoingAttachments,
    };

    const outgoingMessages = [...activeSession.messages, nextUserMessage];

    setSessions((current) =>
      updateSessionById(current, targetSessionId, (session) => ({
        ...session,
        messages: outgoingMessages,
        title: deriveSessionTitle(outgoingMessages),
        updatedAt: Date.now(),
      })),
    );

    setInput("");
    setAttachments([]);
    setIsSending(true);

    const abortController = new AbortController();
    let timeout: number | undefined;

    try {
      timeout = window.setTimeout(() => abortController.abort(), 45_000);
      const response = await fetch(targetUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        signal: abortController.signal,
        body: JSON.stringify({
          message: prompt,
          stream: shouldStream,
          messages: outgoingMessages.map(({ role, content }) => ({ role, content })),
          attachments: outgoingAttachments,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}.`);
      }

      const isEventStream =
        response.headers.get("content-type")?.includes("text/event-stream") ??
        false;

      if (isEventStream && response.body) {
        const assistantId = createMessageId("assistant");
        let streamedError: string | null = null;

        setSessions((current) =>
          updateSessionById(current, targetSessionId, (session) => {
            const messages = [
              ...session.messages,
              { id: assistantId, role: "assistant", content: "" } satisfies ChatMessage,
            ];

            return {
              ...session,
              messages,
              title: deriveSessionTitle(messages),
              updatedAt: Date.now(),
            };
          }),
        );

        const result = await readSseStream(response.body, {
          onDelta: (delta) => {
            const safeDelta = sanitizeAssistantText(delta);
            if (!safeDelta) return false;

            setSessions((current) =>
              updateSessionById(current, targetSessionId, (session) => {
                const messages = session.messages.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content + safeDelta }
                    : message,
                );

                return {
                  ...session,
                  messages,
                  updatedAt: Date.now(),
                };
              }),
            );

            return true;
          },
          onModel: (model) => {
            setSessions((current) =>
              updateSessionById(current, targetSessionId, (session) => ({
                ...session,
                activeModel: model,
                updatedAt: Date.now(),
              })),
            );
          },
          onError: (streamError) => {
            streamedError = streamError;
          },
        });

        if (streamedError) {
          throw new Error(streamedError);
        }

        if (!result.hasText) {
          throw new Error("API returned an empty stream.");
        }
      } else {
        const isJsonResponse =
          response.headers.get("content-type")?.includes("application/json") ??
          false;

        const payload: unknown = isJsonResponse
          ? await response.json()
          : await response.text();

        const reply = sanitizeAssistantText(extractAssistantReply(payload).trim());

        if (!reply) {
          throw new Error("API returned an empty response.");
        }

        const returnedModel = extractModel(payload);

        setSessions((current) =>
          updateSessionById(current, targetSessionId, (session) => {
            const messages = [
              ...session.messages,
              {
                id: createMessageId("assistant"),
                role: "assistant",
                content: reply,
              } satisfies ChatMessage,
            ];

            return {
              ...session,
              messages,
              activeModel: returnedModel || session.activeModel,
              updatedAt: Date.now(),
            };
          }),
        );
      }
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : "Unknown API error.";

      setSessions((current) =>
        updateSessionById(current, targetSessionId, (session) => {
          const messages = [
            ...session.messages,
            {
              id: createMessageId("system"),
              role: "system",
              content: `Request failed: ${message}`,
            } satisfies ChatMessage,
          ];

          return {
            ...session,
            messages,
            updatedAt: Date.now(),
          };
        }),
      );

      setError(message);
    } finally {
      if (timeout !== undefined) {
        window.clearTimeout(timeout);
      }
      setIsSending(false);
      requestAnimationFrame(() => {
        messageInputRef.current?.focus();
      });
    }
  };

  const activeMessages = activeSession?.messages ?? [];

  return (
    <SidebarProvider defaultOpen>
      <Sidebar
        collapsible="offcanvas"
        variant="sidebar"
        className="border-r border-sidebar-border bg-sidebar"
      >
        <SidebarHeader className="gap-2 px-3 py-3">
          <Button
            variant="default"
            size="default"
            className="h-9 justify-start bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90"
            onClick={createNewChat}
          >
            New chat
          </Button>
        </SidebarHeader>

        <SidebarSeparator />

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Recent chats</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {hasLoadedState
                  ? orderedSessions.map((session) => (
                      <SidebarMenuItem key={session.id}>
                        <SidebarMenuButton
                          isActive={session.id === activeSession?.id}
                          onClick={() => {
                            setActiveSessionId(session.id);
                            setError(null);
                          }}
                          className="h-auto items-start py-2"
                        >
                          <div className="flex w-full flex-col gap-0.5 text-left">
                            <span className="truncate text-xs text-foreground">
                              {session.title}
                            </span>
                            <span className="text-[10px] text-muted-foreground">
                              {formatRelativeTime(session.updatedAt)}
                            </span>
                          </div>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))
                  : Array.from({ length: 6 }).map((_, index) => (
                      <SidebarMenuItem key={`sidebar-skeleton-${index}`}>
                        <div className="rounded-md px-2 py-2">
                          <Skeleton className="h-3 w-[84%]" />
                          <Skeleton className="mt-1 h-2 w-[34%]" />
                        </div>
                      </SidebarMenuItem>
                    ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter className="px-3 pb-3 pt-0">
          <Button
            type="button"
            variant="outline"
            size="default"
            className="h-9 justify-start"
            render={<Link href="/settings" />}
          >
            Settings
          </Button>
        </SidebarFooter>

        <SidebarRail />
      </Sidebar>

      <SidebarInset className="bg-background">
        <main className="flex min-h-screen flex-col">
          <header className="sticky top-0 z-20 border-b border-border bg-background/95 backdrop-blur">
            <div className="mx-auto flex w-full max-w-4xl items-center gap-3 px-4 py-3">
              <MobileSidebarToggle />
              <div className="min-w-0 flex-1">
                <p className="truncate text-xl font-semibold text-foreground">AgroLens</p>
              </div>
            </div>
          </header>

          <section
            ref={listRef}
            className="chat-scroll mx-auto mt-4 flex w-full max-w-4xl flex-1 flex-col gap-4 overflow-y-auto px-4 pb-5"
          >
            {!hasLoadedState
              ? Array.from({ length: 3 }).map((_, index) => (
                  <article
                    key={`message-skeleton-${index}`}
                    className={`flex ${index % 2 === 0 ? "justify-start" : "justify-end"}`}
                  >
                    <div className="w-full max-w-2xl rounded-md border border-border bg-card p-4">
                      <Skeleton className="h-3 w-20" />
                      <Skeleton className="mt-3 h-3 w-[92%]" />
                      <Skeleton className="mt-2 h-3 w-[85%]" />
                      <Skeleton className="mt-2 h-3 w-[62%]" />
                    </div>
                  </article>
                ))
              : activeMessages.map((message) => {
                  const isUser = message.role === "user";
                  const isSystem = message.role === "system";
                  const isAssistant = message.role === "assistant";

                  return (
                    <article
                      key={message.id}
                      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={
                          isUser
                            ? "ml-auto w-full max-w-[min(42rem,82%)]"
                            : "w-full max-w-2xl"
                        }
                      >
                        {isAssistant ? (
                          <div className="streamdown-message px-1 text-sm leading-relaxed text-foreground">
                            <Streamdown mode="static" parseIncompleteMarkdown={false}>
                              {message.content}
                            </Streamdown>
                          </div>
                        ) : (
                          <>
                            {message.attachments && message.attachments.length > 0 ? (
                              <ul
                                className={`mb-2 flex gap-2 overflow-x-auto pb-1 ${
                                  isUser ? "justify-end" : ""
                                }`}
                              >
                                {message.attachments.map((attachment, index) => (
                                  <li
                                    key={`${message.id}-attachment-${index}`}
                                    className="w-28 shrink-0 rounded-md border border-border bg-background p-1.5"
                                  >
                                    <div className="relative h-16 w-full overflow-hidden rounded-md">
                                      <Image
                                        src={attachment.dataUrl}
                                        alt={attachment.name}
                                        fill
                                        unoptimized
                                        sizes="112px"
                                        className="object-cover"
                                      />
                                    </div>
                                    <p className="mt-1 truncate text-[10px] text-muted-foreground">
                                      {attachment.name}
                                    </p>
                                  </li>
                                ))}
                              </ul>
                            ) : null}

                            <div
                              className={`rounded-md border ${
                                isSystem
                                  ? "border-destructive/40 bg-destructive/10 p-4"
                                  : "border-border bg-card px-3 py-2 w-fit max-w-full ml-auto"
                              }`}
                            >
                              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground">
                                {message.content}
                              </p>
                            </div>
                          </>
                        )}
                      </div>
                    </article>
                  );
                })}

            {isSending ? (
              <article className="flex justify-start">
                <div className="w-full max-w-2xl px-1 py-2">
                  <p className="text-sm text-muted-foreground">Assistant is responding</p>
                </div>
              </article>
            ) : null}
          </section>

          <footer className="sticky bottom-0 z-20 bg-background/95 backdrop-blur">
            <form onSubmit={submitPrompt} className="mx-auto w-full max-w-4xl px-4 py-3">
              <div className="rounded-md border border-border bg-card p-3">
                {attachments.length > 0 ? (
                  <ul className="mb-3 flex gap-2 overflow-x-auto pb-1">
                    {attachments.map((attachment) => (
                      <li
                        key={attachment.id}
                        className="w-28 shrink-0 rounded-md border border-border bg-background p-1.5"
                      >
                        <div className="relative h-14 w-full overflow-hidden rounded-md">
                          <Image
                            src={attachment.dataUrl}
                            alt={attachment.name}
                            fill
                            unoptimized
                            sizes="112px"
                            className="object-cover"
                          />
                        </div>
                        <div className="mt-1 flex items-center justify-between gap-1">
                          <p className="truncate text-[10px] text-muted-foreground">
                            {attachment.name}
                          </p>
                          <Button
                            type="button"
                            variant="ghost"
                            size="xs"
                            onClick={() =>
                              setAttachments((current) =>
                                current.filter((item) => item.id !== attachment.id),
                              )
                            }
                            className="h-5 px-1 text-[10px] text-muted-foreground"
                          >
                            Remove
                          </Button>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : null}

                <div className="flex items-center gap-2">
                  <Input
                    id="message"
                    name="message"
                    ref={messageInputRef}
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder="Describe the crop issue and ask your question."
                    autoComplete="off"
                    className="h-9 bg-background"
                  />

                  <Button
                    type="submit"
                    size="default"
                    className="h-9"
                    disabled={isSending}
                  >
                    {isSending ? "Sending" : "Send"}
                  </Button>
                </div>

                <div
                  onDragOver={(event) => {
                    event.preventDefault();
                    if (!isDraggingFiles) setIsDraggingFiles(true);
                  }}
                  onDragLeave={(event) => {
                    event.preventDefault();
                    if (
                      event.relatedTarget === null ||
                      !event.currentTarget.contains(event.relatedTarget as Node)
                    ) {
                      setIsDraggingFiles(false);
                    }
                  }}
                  onDrop={onDropFiles}
                  className={`mt-2 rounded-md border border-dashed px-3 py-2 text-xs ${
                    isDraggingFiles
                      ? "border-primary/70 bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground"
                  }`}
                >
                  Drop images here
                </div>

                <div className="mt-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
                  <p>
                    {error ? `Error: ${error}` : `${input.length}/${MAX_MESSAGE_CHARS}`}
                  </p>
                  <p>
                    {attachments.length}/{MAX_ATTACHMENTS} images
                  </p>
                </div>
              </div>
            </form>
          </footer>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
