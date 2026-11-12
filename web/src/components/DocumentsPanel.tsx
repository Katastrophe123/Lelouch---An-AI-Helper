import { useEffect, useRef, useState } from "react";
import { bytes } from "../format";
import type { DocumentInfo } from "../types";
import { Icon } from "./Icon";

interface Props {
  open: boolean;
  documents: DocumentInfo[];
  uploading: string[];
  onClose: () => void;
  onUpload: (files: File[]) => void;
  onDelete: (doc: DocumentInfo) => void;
}

const ACCEPT = ".pdf,.docx,.txt,.md,.markdown,.csv,.tsv,.json,.yaml,.yml,.html,.xml,.py,.js,.ts,.java,.c,.cpp,.go,.rs,.sql,.tex,.log,.rst";

export function DocumentsPanel({ open, documents, uploading, onClose, onUpload, onDelete }: Props) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeBtn.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  const chunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);

  return (
    <div className="drawer-scrim" onClick={onClose}>
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="docs-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="drawer-head">
          <div>
            <h2 id="docs-title">Your documents</h2>
            <p className="drawer-sub">
              Agent chats search these before answering and name the file they used.
            </p>
          </div>
          <button ref={closeBtn} type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <Icon name="close" />
          </button>
        </header>

        <button
          type="button"
          className={`dropzone${over ? " is-over" : ""}`}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const files = Array.from(e.dataTransfer.files);
            if (files.length) onUpload(files);
          }}
        >
          <Icon name="upload" size={22} />
          <span className="dropzone-title">Drop files here or choose them</span>
          <span className="dropzone-sub">PDF, Word, text, Markdown, CSV or code, up to 20 MB each</span>
        </button>
        <input
          ref={input}
          type="file"
          multiple
          hidden
          accept={ACCEPT}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            if (files.length) onUpload(files);
            e.target.value = "";
          }}
        />

        <ul className="doc-list">
          {uploading.map((name) => (
            <li key={`up-${name}`} className="doc-row is-uploading">
              <Icon name="docs" />
              <div className="doc-info">
                <span className="doc-name">{name}</span>
                <span className="doc-sub">Reading and indexing</span>
              </div>
            </li>
          ))}
          {documents.map((doc) => (
            <li key={doc.id} className="doc-row">
              <Icon name="docs" />
              <div className="doc-info">
                <span className="doc-name" title={doc.filename}>
                  {doc.filename}
                </span>
                <span className="doc-sub">
                  {bytes(doc.size_bytes)}, {doc.chunk_count} {doc.chunk_count === 1 ? "passage" : "passages"}
                </span>
              </div>
              <button
                type="button"
                className="icon-btn danger"
                onClick={() => onDelete(doc)}
                aria-label={`Delete ${doc.filename}`}
                title="Delete"
              >
                <Icon name="trash" size={16} />
              </button>
            </li>
          ))}
        </ul>
        {!documents.length && !uploading.length && (
          <p className="drawer-empty">Nothing here yet. Add notes, papers or reports and ask about them in an agent chat.</p>
        )}
        {documents.length > 0 && (
          <p className="drawer-foot">
            {documents.length} {documents.length === 1 ? "file" : "files"} split into {chunks} searchable{" "}
            {chunks === 1 ? "passage" : "passages"}.
          </p>
        )}
      </aside>
    </div>
  );
}
