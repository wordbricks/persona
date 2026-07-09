import { describe, expect, it } from "vitest";

import {
  chunkPersonaSourceText,
  formatPersonaSourceExcerptForQuery,
  hashPersonaSourceContent,
} from "./source-ingestion";

describe("persona source ingestion helpers", () => {
  it("chunks long source text with bounded overlap", () => {
    const text = Array.from(
      { length: 12 },
      (_, index) =>
        `Paragraph ${index} says Mira asks for evidence before accepting confidence.`
    ).join("\n\n");

    const chunks = chunkPersonaSourceText(text, {
      maxChars: 400,
      overlapChars: 30,
    });

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.every((chunk) => chunk.text.length <= 400)).toBe(true);
    for (const [index, chunk] of chunks.entries()) {
      expect(chunk.startChar).toBeLessThan(chunk.endChar);
      if (index > 0) {
        const previous = chunks.at(index - 1);
        expect(previous).toBeDefined();
        expect(chunk.startChar).toBeLessThan(previous?.endChar ?? 0);
      }
    }
  });

  it("uses stable SHA-256 content hashes for source dedupe", async () => {
    await expect(hashPersonaSourceContent("same source")).resolves.toBe(
      await hashPersonaSourceContent("same source")
    );
    await expect(hashPersonaSourceContent("same source")).resolves.not.toBe(
      await hashPersonaSourceContent("different source")
    );
  });

  it("summarizes source excerpts around the retrieval query", () => {
    const excerpt = formatPersonaSourceExcerptForQuery({
      maxChars: 160,
      query: "칫솔 브랜드 사용 제품",
      text: [
        "앞부분은 브랜드 전략과 매장 운영에 대한 긴 대화다. ".repeat(8),
        "중간에도 메뉴 구성과 고객 경험을 계속 이야기한다. ".repeat(8),
        "영상에서 한서린은 칫솔을 고를 때 세라딘 브랜드를 쓴다고 말한다.",
        "마지막에는 선택 기준과 디테일의 중요성을 설명한다.",
      ].join(" "),
    });

    expect(excerpt).toContain("세라딘");
    expect(excerpt).toContain("칫솔");
    expect(excerpt.length).toBeLessThanOrEqual(164);
  });
});
