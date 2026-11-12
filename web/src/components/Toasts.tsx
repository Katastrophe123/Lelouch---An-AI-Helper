export interface Toast {
  id: number;
  text: string;
  tone: "info" | "error";
}

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <button key={toast.id} type="button" className="toast" data-tone={toast.tone} onClick={() => onDismiss(toast.id)}>
          {toast.text}
        </button>
      ))}
    </div>
  );
}
