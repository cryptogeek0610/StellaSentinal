/**
 * DOM Path Utilities
 * 
 * Functions to generate and display DOM paths for debugging and development.
 */

export interface DOMPathInfo {
  path: string;
  position: {
    top: number;
    left: number;
    width: number;
    height: number;
  };
  element: HTMLElement;
  reactComponent?: string;
  htmlElement: string;
}

/**
 * Generates a CSS selector path for a given element
 */
export function getDOMPath(element: HTMLElement | null): string {
  if (!element) return '';

  const path: string[] = [];
  let current: HTMLElement | null = element;

  while (current && current !== document.body && current !== document.documentElement) {
    let selector = current.tagName.toLowerCase();

    // Add ID if present
    if (current.id) {
      selector += `#${current.id}`;
      path.unshift(selector);
      break; // IDs are unique, so we can stop here
    }

    // Add classes if present
    if (current.className && typeof current.className === 'string') {
      const classes = current.className
        .split(' ')
        .filter((c) => c.trim())
        .map((c) => `.${c.replace(/\s+/g, '.')}`)
        .join('');
      if (classes) {
        selector += classes;
      }
    }

    // Add nth-child if needed for uniqueness
    const parent = current.parentElement;
    if (parent) {
      const siblings = Array.from(parent.children).filter(
        (child) => child.tagName === current!.tagName
      );
      if (siblings.length > 1) {
        const index = siblings.indexOf(current) + 1;
        selector += `:nth-of-type(${index})`;
      }
    }

    path.unshift(selector);
    current = current.parentElement;
  }

  return path.join(' > ');
}

/**
 * Gets comprehensive DOM path information including position and element details
 */
export function getDOMPathInfo(element: HTMLElement | null): DOMPathInfo | null {
  if (!element) return null;

  const rect = element.getBoundingClientRect();
  const path = getDOMPath(element);

  // Try to detect React component name from data attributes or fiber
  let reactComponent: string | undefined;
  const reactFiberKey = Object.keys(element).find((key) => key.startsWith('__reactFiber'));
  if (reactFiberKey) {
    const fiber = (element as any)[reactFiberKey];
    if (fiber?.return?.type) {
      const componentName = fiber.return.type.displayName || fiber.return.type.name;
      if (componentName) {
        reactComponent = componentName;
      }
    }
  }

  // Get HTML element tag
  const htmlElement = `<${element.tagName.toLowerCase()}${element.className ? ` class="${element.className}"` : ''}${element.id ? ` id="${element.id}"` : ''}>`;

  return {
    path,
    position: {
      top: Math.round(rect.top),
      left: Math.round(rect.left),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    },
    element,
    reactComponent,
    htmlElement,
  };
}

/**
 * Formats DOM path info as a readable string
 */
export function formatDOMPathInfo(info: DOMPathInfo): string {
  const parts = [
    `DOM Path: ${info.path}`,
    `Position: top=${info.position.top}px, left=${info.position.left}px, width=${info.position.width}px, height=${info.position.height}px`,
  ];

  if (info.reactComponent) {
    parts.push(`React Component: ${info.reactComponent}`);
  }

  parts.push(`HTML Element: ${info.htmlElement}`);

  return parts.join('\n');
}

/**
 * Logs DOM path info to console
 */
export function logDOMPath(element: HTMLElement | null): void {
  const info = getDOMPathInfo(element);
  if (info) {
    console.log(formatDOMPathInfo(info));
    console.log('Element:', element);
  }
}

/**
 * Global function to enable/disable automatic DOM path logging
 * Call this from browser console: window.enableDOMPathLogging(true)
 */
export function setupGlobalDOMPathLogging(): void {
  if (typeof window !== 'undefined') {
    (window as any).enableDOMPathLogging = (enabled: boolean) => {
      (window as any).__DOM_PATH_LOGGING_ENABLED__ = enabled;
      console.log(`DOM Path logging ${enabled ? 'enabled' : 'disabled'}`);
    };

    (window as any).logDOMPath = (selector: string) => {
      const element = document.querySelector(selector) as HTMLElement;
      if (element) {
        logDOMPath(element);
      } else {
        console.warn(`Element not found: ${selector}`);
      }
    };

    (window as any).logAllDOMPaths = () => {
      const elements = document.querySelectorAll('[data-cursor-element-id]');
      console.log(`\n=== Found ${elements.length} elements with data-cursor-element-id ===\n`);
      elements.forEach((el, index) => {
        const info = getDOMPathInfo(el as HTMLElement);
        if (info) {
          console.log(`\n--- Element ${index + 1} ---`);
          console.log(formatDOMPathInfo(info));
        }
      });
    };

    console.log('DOM Path utilities available:');
    console.log('  - window.enableDOMPathLogging(true/false) - Enable/disable auto logging');
    console.log('  - window.logDOMPath("selector") - Log path for element');
    console.log('  - window.logAllDOMPaths() - Log all elements with data-cursor-element-id');
  }
}

// Note: React hook removed - use getDOMPathInfo directly with ref.current

