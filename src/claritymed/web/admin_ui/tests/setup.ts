import "@testing-library/jest-dom/vitest";

// jsdom does not implement matchMedia; Mantine's color-scheme hook
// reads it on mount. Polyfill with a no-op so component tests render
// without throwing.
if (typeof window !== "undefined" && !window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }),
    writable: true,
  });
}
