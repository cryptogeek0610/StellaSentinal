/**
 * useKeyboardShortcuts - Power user keyboard navigation
 *
 * Steve Jobs principle: Power users deserve love too.
 * J/K navigation, F to fix, E to expand - like vim for investigations.
 */

import { useEffect, useCallback } from 'react';

interface KeyboardShortcutsConfig {
  onNext?: () => void;
  onPrevious?: () => void;
  onExpand?: () => void;
  onFix?: () => void;
  onEscape?: () => void;
  enabled?: boolean;
}

export function useKeyboardShortcuts({
  onNext,
  onPrevious,
  onExpand,
  onFix,
  onEscape,
  enabled = true,
}: KeyboardShortcutsConfig) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      switch (event.key.toLowerCase()) {
        case 'j':
          event.preventDefault();
          onNext?.();
          break;
        case 'k':
          event.preventDefault();
          onPrevious?.();
          break;
        case 'e':
        case 'enter':
          if (!event.metaKey && !event.ctrlKey) {
            event.preventDefault();
            onExpand?.();
          }
          break;
        case 'f':
          if (!event.metaKey && !event.ctrlKey) {
            event.preventDefault();
            onFix?.();
          }
          break;
        case 'escape':
          event.preventDefault();
          onEscape?.();
          break;
      }
    },
    [onNext, onPrevious, onExpand, onFix, onEscape]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown, enabled]);
}

export default useKeyboardShortcuts;
