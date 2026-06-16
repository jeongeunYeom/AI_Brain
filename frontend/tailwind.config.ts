import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        petroleum: {
          950: "#071311",
          900: "#0d1f1b",
          700: "#1f4d3f",
          400: "#52c39b",
          300: "#8ee6c1"
        }
      }
    }
  },
  plugins: []
};

export default config;
