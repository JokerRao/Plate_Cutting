'use client';

import { useEffect } from 'react';

interface UnsavedChangesPromptProps {
  hasChanges: boolean;
}

/**
 * 关闭/刷新标签页时的浏览器级提示。不在 document 上捕获链接点击，避免与 Next.js 路由、复杂 DOM 冲突。
 * 应用内离开请使用各页的按钮逻辑（如 handleBack）配合 useAppDialog。
 */
export default function UnsavedChangesPrompt({ hasChanges }: UnsavedChangesPromptProps) {
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasChanges) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [hasChanges]);

  return null;
}
