import { useState } from "react";
import { describeStep, seconds } from "../format";
import type { Step } from "../types";

interface Props {
  steps: Step[];
  /** True while the turn is still running: the rail is open and the last step pulses. */
  live?: boolean;
}

export function ActivityRail({ steps, live = false }: Props) {
  const [open, setOpen] = useState(false);
  const visible = steps.filter((s) => s.node !== "answer" || s.status === "error");
  if (!visible.length && !live) return null;

  const expanded = live || open;
  const total = steps.reduce((sum, s) => sum + (s.latency_ms || 0), 0);
  const failures = visible.filter((s) => s.status === "error").length;
  const summary = `${visible.length} ${visible.length === 1 ? "step" : "steps"}${
    total ? `, ${seconds(total)}` : ""
  }${failures ? `, ${failures} with problems` : ""}`;

  return (
    <section className={`rail${live ? " is-live" : ""}`} aria-label="What the assistant did">
      {!live && (
        <button type="button" className="rail-toggle" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          <span className="rail-toggle-chevron" data-open={open} aria-hidden="true">
            ›
          </span>
          {open ? "Hide the work" : `Show the work: ${summary}`}
        </button>
      )}
      {expanded && (
        <ol className="rail-list">
          {visible.map((step, index) => {
            const view = describeStep(step);
            const current = live && index === visible.length - 1;
            return (
              <li
                key={step.sequence}
                className="rail-step"
                data-kind={view.kind}
                data-status={step.status}
                data-current={current || undefined}
              >
                <span className="rail-marker" aria-hidden="true" />
                <div className="rail-body">
                  <span className="rail-title">{view.title}</span>
                  {view.detail && <span className="rail-detail">{view.detail}</span>}
                  {step.status === "error" && step.error && step.node !== "answer" && (
                    <span className="rail-error">{step.error}</span>
                  )}
                </div>
              </li>
            );
          })}
          {live && !visible.length && (
            <li className="rail-step" data-current>
              <span className="rail-marker" aria-hidden="true" />
              <div className="rail-body">
                <span className="rail-title">Starting up</span>
              </div>
            </li>
          )}
        </ol>
      )}
    </section>
  );
}
