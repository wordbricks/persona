import { describe, expect, it } from "vitest";

import {
  hashPersonaMemoryText,
  normalizePersonaEmbeddingText,
} from "./embeddings";

describe("normalizePersonaEmbeddingText", () => {
  it("trims and caps oversized memory text", () => {
    expect(normalizePersonaEmbeddingText("  hello  ")).toBe("hello");
    const oversized = "가".repeat(10_000);
    expect(normalizePersonaEmbeddingText(oversized)).toHaveLength(6_000);
  });
});

describe("hashPersonaMemoryText", () => {
  it("is stable for identical text and differs across texts", () => {
    expect(hashPersonaMemoryText("단순함이 최고다")).toBe(
      hashPersonaMemoryText("단순함이 최고다")
    );
    expect(hashPersonaMemoryText("단순함이 최고다")).not.toBe(
      hashPersonaMemoryText("복잡함이 최고다")
    );
    expect(hashPersonaMemoryText("simplicity")).toMatch(/^[0-9a-f]{8}$/);
  });
});
