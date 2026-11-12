import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Api, ApiError } from "../api";
import { modelName } from "../format";
import type { AgentConfig, Conversation, ConversationSummary, DocumentInfo, Health, Mode, Session, Step } from "../types";
import { Composer } from "./Composer";
import { DocumentsPanel } from "./DocumentsPanel";
import { Icon } from "./Icon";
import { AssistantMessage, LiveAssistant, UserMessage, type LiveTurn } from "./MessageView";
import { Toasts, type Toast } from "./Toasts";
import { Sidebar } from "./Sidebar";

const SUGGESTIONS: Record<Mode, string[]> = {
  agent: [
    "Find three recent, reliable sources on solid-state batteries and compare what they claim",
    "Open news.ycombinator.com and summarise the five top stories in one line each",
    "Summarise the documents I've uploaded and list the questions they leave open",
    "Plan a two-week revision timetable for my exams, one subject per day",
  ],
  tutor: [
    "What is 17 × 23? Show the working.",
    "A ball is dropped from 20 m. How long until it hits the ground?",
    "What is 15% of 240?",
  ],
};

function readHash(): string | null {
  const match = /^#\/c\/([0-9a-f-]{36})$/.exec(window.location.hash);
  return match?.[1] ?? null;
}

export function Workspace({ session, onSignOut }: { session: Session; onSignOut: () => void }) {
  const api = useMemo(() => new Api(session.access_token), [session.access_token]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(readHash);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [uploading, setUploading] = useState<string[]>([]);
  const [docsOpen, setDocsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [draftMode, setDraftMode] = useState<Mode>("agent");
  const [draftModel, setDraftModel] = useState("");
  const [live, setLive] = useState<LiveTurn | null>(null);
  const [prefill, setPrefill] = useState<{ text: string; n: number }>();
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const toast = useCallback((text: string, tone: Toast["tone"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((all) => [...all, { id, text, tone }]);
    setTimeout(() => setToasts((all) => all.filter((t) => t.id !== id)), 5000);
  }, []);

  const handle = useCallback(
    (err: unknown, fallback: string) => {
      if (err instanceof ApiError && err.status === 401) {
        onSignOut();
        return;
      }
      toast(err instanceof ApiError ? err.message : fallback, "error");
    },
    [onSignOut, toast],
  );

  const refreshList = useCallback(async () => {
    try {
      setConversations((await api.listConversations()).items);
    } catch (err) {
      handle(err, "Couldn't load your chats.");
    }
  }, [api, handle]);

  const refreshDocs = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
    } catch (err) {
      handle(err, "Couldn't load your documents.");
    }
  }, [api, handle]);

  // Initial load.
  useEffect(() => {
    void refreshList();
    void refreshDocs();
    api.health().then(setHealth, () => setHealth(null));
    api.agentConfig().then(
      (cfg) => {
        setConfig(cfg);
        setDraftModel(cfg.default_model);
      },
      () => setConfig(null),
    );
  }, [api, refreshList, refreshDocs]);

  // Keep the URL in step with the open chat, so reload and back/forward work.
  useEffect(() => {
    const target = activeId ? `#/c/${activeId}` : "#/";
    if (window.location.hash !== target) window.history.pushState(null, "", target);
  }, [activeId]);
  useEffect(() => {
    const onPop = () => setActiveId(readHash());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // Load the open chat.
  useEffect(() => {
    if (!activeId) {
      setConversation(null);
      return;
    }
    let cancelled = false;
    api.getConversation(activeId).then(
      (c) => !cancelled && setConversation(c),
      (err) => {
        if (cancelled) return;
        handle(err, "Couldn't open that chat.");
        setActiveId(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [activeId, api, handle]);

  // Follow the conversation as it grows, unless the reader scrolled up.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    if (nearBottom || live?.steps.length === 0) el.scrollTop = el.scrollHeight;
  }, [conversation, live]);

  const upload = useCallback(
    async (files: File[]) => {
      setDocsOpen(true);
      for (const file of files) {
        setUploading((names) => [...names, file.name]);
        try {
          await api.uploadDocument(file);
          toast(`Added ${file.name}`);
        } catch (err) {
          handle(err, `Couldn't add ${file.name}.`);
        } finally {
          setUploading((names) => names.filter((n) => n !== file.name));
        }
      }
      await refreshDocs();
    },
    [api, handle, refreshDocs, toast],
  );

  const removeDoc = async (doc: DocumentInfo) => {
    if (!window.confirm(`Delete ${doc.filename}? The agent will no longer be able to search it.`)) return;
    try {
      await api.deleteDocument(doc.id);
      toast(`Deleted ${doc.filename}`);
      await refreshDocs();
    } catch (err) {
      handle(err, "Couldn't delete that document.");
    }
  };

  const send = async (text: string) => {
    let id = activeId;
    if (!id) {
      try {
        const created = await api.createConversation(draftMode, draftMode === "agent" ? draftModel : undefined);
        id = created.id;
        setConversation(created);
        setActiveId(id);
        void refreshList();
      } catch (err) {
        handle(err, "Couldn't start a chat.");
        return;
      }
    }
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    const turn: LiveTurn = { question: text, steps: [], text: "" };
    setLive(turn);

    const upsert = (steps: Step[], step: Step) => {
      const index = steps.findIndex((s) => s.sequence === step.sequence);
      if (index === -1) return [...steps, step];
      const next = steps.slice();
      next[index] = step;
      return next;
    };

    try {
      for await (const event of api.ask(id, text, controller.signal)) {
        if (event.name === "step") setLive((t) => t && { ...t, steps: upsert(t.steps, event.data) });
        else if (event.name === "token") setLive((t) => t && { ...t, text: t.text + event.data.text });
        else if (event.name === "done") setLive((t) => t && { ...t, text: event.data.content });
        else if (event.name === "error") setLive((t) => t && { ...t, error: event.data.message });
      }
    } catch (err) {
      if (controller.signal.aborted) {
        toast("Stopped. Anything already finished is saved.");
      } else {
        const message = err instanceof ApiError ? err.message : "Lost the connection to the server.";
        setLive((t) => t && { ...t, error: message });
        if (err instanceof ApiError && err.status === 401) onSignOut();
      }
    } finally {
      abort.current = null;
      setRunning(false);
    }

    try {
      setConversation(await api.getConversation(id));
      setLive((t) => (t?.error ? t : null));
    } catch {
      /* keep the live view */
    }
    void refreshList();
  };

  const stop = () => abort.current?.abort();

  const startNew = () => {
    stop();
    setLive(null);
    setActiveId(null);
    setSidebarOpen(false);
  };

  const mode: Mode = conversation?.mode ?? draftMode;
  const messages = conversation?.messages ?? [];
  // While a turn runs, the stored chat does not include the new question yet.
  const last = messages[messages.length - 1];
  const showLiveQuestion = Boolean(live) && !(last?.role === "user" && last.content === live?.question);
  const empty = !messages.length && !live;

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeId}
        documentCount={documents.length}
        username={session.username}
        health={health}
        onNew={startNew}
        onSelect={(id) => {
          setLive(null);
          setActiveId(id);
          setSidebarOpen(false);
        }}
        onDocuments={() => {
          setDocsOpen(true);
          setSidebarOpen(false);
        }}
        onSignOut={onSignOut}
        onClose={() => setSidebarOpen(false)}
      />

      <main
        className="main"
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes("Files")) {
            e.preventDefault();
            setDragging(true);
          }
        }}
        onDragLeave={(e) => {
          if (e.currentTarget === e.target) setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const files = Array.from(e.dataTransfer.files);
          if (files.length) void upload(files);
        }}
      >
        <header className="topbar">
          <button type="button" className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open chats">
            <Icon name="menu" />
          </button>
          <h1 className="topbar-title">
            {conversation && conversation.title !== "New conversation"
              ? conversation.title
              : (live?.question ?? "New chat")}
          </h1>
          {conversation && (
            <span className="engine" data-mode={conversation.mode}>
              {conversation.mode === "agent" ? `Agent, ${modelName(conversation.agent_model)}` : "Tutor"}
            </span>
          )}
        </header>

        <div className="scroller" ref={scroller}>
          <div className="thread">
            {empty && (
              <section className="welcome">
                <h2 className="welcome-title">
                  {mode === "agent" ? "What shall we take on?" : "Which problem are we solving?"}
                </h2>
                <p className="welcome-sub">
                  {mode === "agent"
                    ? "I can search the web, work a real browser, use your connected tools and read the documents you add."
                    : "I solve it, check my own answer, and only then explain it. If I can't verify it, I say so."}
                </p>
                <ul className="suggestions">
                  {SUGGESTIONS[mode].map((s) => (
                    <li key={s}>
                      <button type="button" onClick={() => setPrefill((p) => ({ text: s, n: (p?.n ?? 0) + 1 }))}>
                        {s}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {messages.map((message) =>
              message.role === "user" ? (
                <UserMessage key={message.id} content={message.content} />
              ) : (
                <AssistantMessage key={message.id} message={message} model={conversation?.agent_model} />
              ),
            )}
            {live && (
              <>
                {showLiveQuestion && <UserMessage content={live.question} />}
                <LiveAssistant turn={live} />
              </>
            )}
          </div>
        </div>

        <div className="composer-wrap">
          <Composer
            busy={running}
            mode={mode}
            prefill={prefill?.text}
            prefillKey={prefill?.n}
            onSend={send}
            onStop={stop}
            onAttach={upload}
            picker={
              conversation
                ? undefined
                : {
                    mode: draftMode,
                    model: draftModel,
                    config,
                    onMode: setDraftMode,
                    onModel: setDraftModel,
                  }
            }
          />
          <p className="composer-note">
            {mode === "agent"
              ? "Lelouch checks its sources, but can still be wrong. Verify anything important."
              : "Answers marked self-checked passed a second review. Others are labelled."}
          </p>
        </div>

        {dragging && (
          <div className="drop-overlay" aria-hidden="true">
            <p>Drop to add to your documents</p>
          </div>
        )}
      </main>

      <DocumentsPanel
        open={docsOpen}
        documents={documents}
        uploading={uploading}
        onClose={() => setDocsOpen(false)}
        onUpload={upload}
        onDelete={removeDoc}
      />
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((all) => all.filter((t) => t.id !== id))} />
    </div>
  );
}
