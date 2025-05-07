'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface UnsavedChangesPromptProps {
  hasChanges: boolean;
  onSave: () => Promise<void>;
}

export default function UnsavedChangesPrompt({ hasChanges, onSave }: UnsavedChangesPromptProps) {
  const router = useRouter();

  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasChanges) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const link = target.closest('a');
      if (link && !link.getAttribute('href')?.startsWith('/project/')) {
        if (hasChanges) {
          e.preventDefault();
          if (window.confirm('有未保存的更改，是否保存？')) {
            onSave().then(() => {
              window.location.href = link.href;
            });
          } else {
            window.location.href = link.href;
          }
        }
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('click', handleClick);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('click', handleClick);
    };
  }, [hasChanges, onSave]);

  return null;
} 