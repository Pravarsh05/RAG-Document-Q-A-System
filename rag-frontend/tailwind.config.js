/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Design tokens for the RAG console.
        // Two accent colors are meaningful, not decorative: amber traces
        // lexical (BM25) hits, teal traces vector (semantic) hits. When a
        // chunk is surfaced by both signals in the hybrid pipeline, the UI
        // blends them rather than picking one.
        ink: {
          950: "#0A0C11",
          900: "#0F131B",
          850: "#141924",
          800: "#1A2130",
          700: "#242D40",
          600: "#333F58",
        },
        mist: {
          400: "#5C6A85",
          300: "#8894AB",
          200: "#B6BFD1",
          100: "#E4E8F0",
        },
        lexical: {
          DEFAULT: "#F2B705",
          dim: "#8A6A1F",
        },
        vector: {
          DEFAULT: "#2DD4BF",
          dim: "#1F6E68",
        },
        hybrid: "#B98CF2",
        ok: "#34D399",
        warn: "#F2B705",
        err: "#F87171",
      },
      fontFamily: {
        display: ["'IBM Plex Sans'", "sans-serif"],
        body: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "12px",
      },
      boxShadow: {
        "glow-lexical": "0 0 20px -5px rgba(242, 183, 5, 0.25)",
        "glow-vector": "0 0 20px -5px rgba(45, 212, 191, 0.25)",
        "glow-hybrid": "0 0 20px -5px rgba(185, 140, 242, 0.25)",
        "subtle": "0 2px 8px rgba(0, 0, 0, 0.4)",
        "card": "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideInRight: {
          "0%": { opacity: "0", transform: "translateX(20px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
      animation: {
        fadeIn: "fadeIn 0.2s ease-out forwards",
        slideInRight: "slideInRight 0.25s ease-out forwards",
      },
    },
  },
  plugins: [],
};
