import { defineConfig } from "tsup";

export default defineConfig({
  clean: true,
  dts: true,
  entry: {
    "agent/index": "src/agent/index.ts",
    index: "src/index.ts",
    "memory/index": "src/memory/index.ts",
    "schema/index": "src/schema/index.ts",
  },
  format: ["esm"],
  sourcemap: true,
});
