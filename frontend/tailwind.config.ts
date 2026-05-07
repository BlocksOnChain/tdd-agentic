import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0d10",
        surface: "#11151b",
        muted: "#1a1f27",
        border: "#272d36",
        accent: "#5b8def",
      },
    },
  },
  plugins: [],
};

export default config;
