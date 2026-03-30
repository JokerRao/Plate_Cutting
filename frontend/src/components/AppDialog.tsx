'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

type DialogOpen =
  | {
      kind: 'alert';
      message: string;
      title?: string;
      resolve: () => void;
    }
  | {
      kind: 'confirm';
      message: string;
      title?: string;
      resolve: (value: boolean) => void;
    };

export type AppDialogContextValue = {
  alert: (message: string, title?: string) => Promise<void>;
  confirm: (message: string, title?: string) => Promise<boolean>;
};

const AppDialogContext = createContext<AppDialogContextValue | null>(null);

export function useAppDialog(): AppDialogContextValue {
  const ctx = useContext(AppDialogContext);
  if (!ctx) {
    throw new Error('useAppDialog must be used within AppDialogProvider');
  }
  return ctx;
}

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function listFocusables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}

function AppDialogLayer({ open }: { open: DialogOpen | null }) {
  const layerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        if (open.kind === 'confirm') open.resolve(false);
        else open.resolve();
        return;
      }

      if (e.key !== 'Tab') return;
      const layer = layerRef.current;
      if (!layer) return;

      const list = listFocusables(layer);
      if (list.length === 0) return;

      const idx = list.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey) {
        if (idx <= 0) {
          e.preventDefault();
          list[list.length - 1].focus();
        }
      } else if (idx === list.length - 1 || idx === -1) {
        e.preventDefault();
        list[0].focus();
      }
    };

    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const layer = layerRef.current;
    if (!layer) return;
    const list = listFocusables(layer);
    const t = window.requestAnimationFrame(() => {
      const dialogPanel = layer.querySelector<HTMLElement>('[role="dialog"]');
      const firstAction = dialogPanel?.querySelector<HTMLElement>('button');
      (firstAction ?? list[0] ?? dialogPanel ?? layer).focus();
    });
    return () => window.cancelAnimationFrame(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const prevActive = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
      prevActive?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  const onBackdrop = () => {
    if (open.kind === 'confirm') open.resolve(false);
    else open.resolve();
  };

  return (
    <div ref={layerRef} className="fixed inset-0 z-[200]" aria-live="polite">
      <button
        type="button"
        className="absolute inset-0 bg-black/45 backdrop-blur-[1px]"
        aria-label="关闭"
        onClick={onBackdrop}
      />
      <div className="absolute inset-0 flex items-center justify-center p-4 pointer-events-none">
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby={open.title ? 'app-dialog-title' : undefined}
          tabIndex={-1}
          className="pointer-events-auto w-full max-w-md rounded-xl border border-hairline bg-surface px-5 py-4 shadow-[0_24px_60px_rgba(0,0,0,0.22)] outline-none"
          onClick={(e) => e.stopPropagation()}
        >
          {open.title ? (
            <h2 id="app-dialog-title" className="mb-2 text-base font-semibold tracking-tight text-ink">
              {open.title}
            </h2>
          ) : null}
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{open.message}</p>
          <div className="mt-5 flex justify-end gap-2">
            {open.kind === 'confirm' ? (
              <>
                <button
                  type="button"
                  className="btn-gallery-secondary inline-flex h-8 items-center px-3 text-xs shadow-sm"
                  onClick={() => open.resolve(false)}
                >
                  取消
                </button>
                <button
                  type="button"
                  className="btn-gallery-primary inline-flex h-8 items-center px-3 text-xs shadow-sm"
                  onClick={() => open.resolve(true)}
                >
                  确定
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn-gallery-primary inline-flex h-8 items-center px-3 text-xs shadow-sm"
                onClick={() => open.resolve()}
              >
                确定
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function AppDialogProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState<DialogOpen | null>(null);

  const alertFn = useCallback((message: string, title?: string) => {
    return new Promise<void>((resolve) => {
      setOpen({
        kind: 'alert',
        message,
        title,
        resolve: () => {
          resolve();
          setOpen(null);
        },
      });
    });
  }, []);

  const confirmFn = useCallback((message: string, title?: string) => {
    return new Promise<boolean>((resolve) => {
      setOpen({
        kind: 'confirm',
        message,
        title,
        resolve: (v) => {
          resolve(v);
          setOpen(null);
        },
      });
    });
  }, []);

  const value = useMemo(
    () => ({ alert: alertFn, confirm: confirmFn }),
    [alertFn, confirmFn]
  );

  return (
    <AppDialogContext.Provider value={value}>
      {children}
      <AppDialogLayer open={open} />
    </AppDialogContext.Provider>
  );
}
