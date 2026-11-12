import { relativeDay } from "../format";
import type { ConversationSummary, Health } from "../types";
import { Icon, Mark } from "./Icon";

interface Props {
  open: boolean;
  conversations: ConversationSummary[];
  activeId: string | null;
  documentCount: number;
  username: string;
  health: Health | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDocuments: () => void;
  onSignOut: () => void;
  onClose: () => void;
}

function connectionText(health: Health | null): { text: string; ok: boolean } {
  if (!health) return { text: "Connecting to the server", ok: false };
  if (health.database !== "up") return { text: "Database is offline", ok: false };
  switch (health.agent_auth) {
    case "claude_login":
      return { text: "Agent signed in with your Claude plan", ok: true };
    case "api_key":
      return { text: "Agent using an API key", ok: true };
    default:
      return { text: "Agent in offline demo mode", ok: true };
  }
}

export function Sidebar(props: Props) {
  const { open, conversations, activeId, documentCount, username, health } = props;
  const groups = new Map<string, ConversationSummary[]>();
  for (const conversation of conversations) {
    const key = relativeDay(conversation.updated_at);
    groups.set(key, [...(groups.get(key) ?? []), conversation]);
  }
  const connection = connectionText(health);

  return (
    <>
      <div className={`sidebar-scrim${open ? " is-open" : ""}`} onClick={props.onClose} aria-hidden="true" />
      <nav className={`sidebar${open ? " is-open" : ""}`} aria-label="Chats">
        <div className="sidebar-brand">
          <Mark size={30} />
          <span className="wordmark">Lelouch</span>
        </div>

        <button type="button" className="new-chat" onClick={props.onNew}>
          <Icon name="plus" />
          New chat
        </button>

        <div className="chat-list">
          {!conversations.length && <p className="chat-empty">Your chats will appear here.</p>}
          {[...groups.entries()].map(([group, items]) => (
            <section key={group} className="chat-group">
              <h3>{group}</h3>
              <ul>
                {items.map((c) => (
                  <li key={c.id}>
                    <button
                      type="button"
                      className="chat-item"
                      aria-current={c.id === activeId ? "page" : undefined}
                      onClick={() => props.onSelect(c.id)}
                      title={c.title}
                    >
                      <span className="chat-kind" data-mode={c.mode} aria-label={c.mode === "agent" ? "Agent" : "Tutor"} />
                      <span className="chat-title">{c.title === "New conversation" ? "Untitled chat" : c.title}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>

        <button type="button" className="sidebar-docs" onClick={props.onDocuments}>
          <Icon name="docs" />
          <span>Your documents</span>
          <span className="count">{documentCount}</span>
        </button>

        <footer className="sidebar-foot">
          <div className="account">
            <span className="avatar" aria-hidden="true">
              {username.slice(0, 1).toUpperCase()}
            </span>
            <div className="account-text">
              <span className="account-name">{username}</span>
              <span className="account-status" data-ok={connection.ok}>
                {connection.text}
              </span>
            </div>
          </div>
          <button type="button" className="icon-btn" onClick={props.onSignOut} aria-label="Sign out" title="Sign out">
            <Icon name="signout" />
          </button>
        </footer>
      </nav>
    </>
  );
}
