import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

export type ToastVariant = "default" | "success" | "error" | "info";

export interface ToastItem {
  id: string;
  title: string;
  description?: string;
  variant?: ToastVariant;
}

interface ToastContextValue {
  toasts: ToastItem[];
  toast: (item: Omit<ToastItem, "id">) => void;
  dismiss: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const toast = useCallback((item: Omit<ToastItem, "id">) => {
    const id = crypto.randomUUID();
    setToasts(prev => [...prev, { ...item, id }]);
    setTimeout(() => dismiss(id), 4000);
  }, [dismiss]);

  const value = useMemo(() => ({ toasts, toast, dismiss }), [toasts, toast, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport toasts={toasts} dismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}

function Tone({ variant }: { variant?: ToastVariant }) {
  if (variant === "success") return <span className="h-2 w-2 rounded-full bg-primary" />;
  if (variant === "error") return <span className="h-2 w-2 rounded-full bg-destructive" />;
  if (variant === "info") return <span className="h-2 w-2 rounded-full bg-secondary" />;
  return <span className="h-2 w-2 rounded-full bg-primary" />;
}

function ToastViewport({ toasts, dismiss }: { toasts: ToastItem[]; dismiss: (id: string) => void }) {
  return (
    <div
      className="pointer-events-none fixed inset-0 z-9999 flex flex-col items-end gap-3 px-4 py-4 sm:top-4 sm:right-4 sm:left-auto"
      role="status"
      aria-live="polite"
      aria-atomic="false"
    >
      {toasts.map(toast => (
        <div
          key={toast.id}
          className="pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-lg border border-border/60 bg-card p-4 shadow-lg backdrop-blur"
        >
          <Tone variant={toast.variant} />
          <div className="flex-1">
            <p className="text-sm font-semibold text-foreground">{toast.title}</p>
            {toast.description && <p className="text-xs text-muted-foreground mt-1">{toast.description}</p>}
          </div>
          <button
            type="button"
            onClick={() => dismiss(toast.id)}
            className="text-xs text-muted-foreground hover:text-foreground"
            aria-label="Dismiss notification"
          >
            x
          </button>
        </div>
      ))}
    </div>
  );
}
