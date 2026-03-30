'use client';

type Props = {
  message: string | null;
};

/** 轻提示：固定底部居中，配合父级 state + 定时清除 */
export default function GalleryToast({ message }: Props) {
  if (!message) return null;

  return (
    <div
      key={message}
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className="pointer-events-none fixed bottom-6 left-1/2 z-[100] max-w-[min(90vw,22rem)] -translate-x-1/2 rounded-lg border border-hairline bg-surface px-4 py-2.5 text-center text-sm font-medium text-ink shadow-[0_8px_30px_rgba(0,0,0,0.12),0_0_0_1px_rgba(0,0,0,0.04)] animate-fade-in-up"
    >
      {message}
    </div>
  );
}
