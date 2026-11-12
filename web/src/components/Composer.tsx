import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { modelName } from "../format";
import type { AgentConfig, Mode } from "../types";
import { Icon } from "./Icon";

interface Props {
  busy: boolean;
  /** Present only when starting a new chat; an existing chat's engine is fixed. */
  picker?: {
    mode: Mode;
    model: string;
    config: AgentConfig | null;
    onMode: (mode: Mode) => void;
    onModel: (model: string) => void;
  };
  mode: Mode;
  onSend: (text: string) => void;
  onStop: () => void;
  onAttach: (files: File[]) => void;
  prefill?: string;
  /** Changes on every suggestion click, so the same one can be picked twice. */
  prefillKey?: number;
}

export function Composer({ busy, picker, mode, onSend, onStop, onAttach, prefill, prefillKey }: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (prefill) {
      setText(prefill);
      area.current?.focus();
    }
  }, [prefill, prefillKey]);

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const submit = () => {
    const value = text.trim();
    if (value.length < 3 || busy) return;
    onSend(value);
    setText("");
  };

  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      <textarea
        ref={area}
        value={text}
        rows={1}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={
          mode === "agent"
            ? "Ask Lelouch to research, browse a site, or dig into your documents"
            : "Ask a homework question"
        }
        aria-label="Message"
        maxLength={8000}
      />
      <div className="composer-bar">
        <button
          type="button"
          className="icon-btn"
          onClick={() => fileInput.current?.click()}
          title="Add documents"
          aria-label="Add documents"
        >
          <Icon name="paperclip" />
        </button>
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            if (files.length) onAttach(files);
            e.target.value = "";
          }}
        />
        {picker && (
          <div className="picker" role="group" aria-label="Who answers">
            <div className="segmented">
              {(["agent", "tutor"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={picker.mode === value}
                  onClick={() => picker.onMode(value)}
                >
                  {value === "agent" ? "Agent" : "Tutor"}
                </button>
              ))}
            </div>
            {picker.mode === "agent" && picker.config && (
              <select
                className="model-select"
                value={picker.model}
                onChange={(e) => picker.onModel(e.target.value)}
                aria-label="Model"
              >
                {picker.config.models.map((model) => (
                  <option key={model} value={model}>
                    {modelName(model)}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}
        <span className="composer-hint">Shift + Enter for a new line</span>
        {busy ? (
          <button type="button" className="send-btn is-stop" onClick={onStop} aria-label="Stop">
            <Icon name="stop" size={16} />
          </button>
        ) : (
          <button
            type="button"
            className="send-btn"
            onClick={submit}
            disabled={text.trim().length < 3}
            aria-label="Send"
          >
            <Icon name="send" />
          </button>
        )}
      </div>
    </div>
  );
}
