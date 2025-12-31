/**
 * usePageTitle Hook
 *
 * Updates the browser tab title with the current page name.
 * Format: "Page Name | Stella Sentinel"
 */

import { useEffect } from 'react';

export const usePageTitle = (title: string) => {
  useEffect(() => {
    const previousTitle = document.title;
    document.title = title ? `${title} | Stella Sentinel` : 'Stella Sentinel';

    return () => {
      document.title = previousTitle;
    };
  }, [title]);
};
