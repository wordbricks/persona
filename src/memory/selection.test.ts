import { describe, expect, it } from "vitest";

import {
  calculateMemoryAvailability,
  normalizeCosineSimilarity,
  selectPersonaMemories,
  selectPersonaMemoriesWithScores,
} from "./selection";
import type { PersonaMemoryCandidate, PersonaTurnPlan } from "./types";

const turnPlan: PersonaTurnPlan = {
  context: {
    contextKeys: {
      domains: ["product"],
      emotions: [],
      entities: ["iphone"],
      lifeStages: [],
      themes: ["simplicity"],
      timePeriods: [],
    },
    retrievalQueries: {
      episodic: "product launch simplicity",
      habit: "respond to feature creep",
      semantic: "beliefs about simplicity",
      source: "simplicity sources",
      style: "speaking style",
    },
  },
  gate: {
    answerMode: "persona_grounded_response",
    audit: {
      confidence: 0.9,
      matchedSignals: ["asks for persona judgment"],
      requiresClarification: false,
    },
    neededMemoryTypes: ["semantic_belief"],
    queryFocus: {
      emotionHints: [],
      people: [],
      themes: ["simplicity"],
      time: null,
    },
    queryType: "value_judgment",
    safetyFlags: [],
  },
};

function buildCandidate(
  overrides: Partial<PersonaMemoryCandidate> &
    Pick<PersonaMemoryCandidate, "id" | "kind">
): PersonaMemoryCandidate {
  return {
    activationCount: 0,
    affect: null,
    aliasIds: [],
    confidence: 0.8,
    createdAt: new Date("2026-01-01T00:00:00.000Z"),
    emotions: [],
    guidance: [],
    lastActivatedAt: null,
    linkIds: [],
    privacyLevel: "internal",
    retrievalBoost: 1,
    semanticSimilarity: null,
    sourceConfidence: 0.8,
    strength: 0.8,
    summary: "summary",
    text: "unrelated text",
    themes: [],
    voice: null,
    ...overrides,
  };
}

describe("selectPersonaMemories", () => {
  it("keeps identity memory active even when episodic matches dominate", () => {
    const episodes = Array.from({ length: 10 }, (_, index) =>
      buildCandidate({
        id: `episode_${index}`,
        kind: "episode",
        text: "product launch simplicity keynote",
      })
    );
    const style = buildCandidate({
      id: "style_1",
      kind: "style",
      strength: 0.9,
    });
    const belief = buildCandidate({
      id: "belief_1",
      kind: "belief",
      strength: 0.95,
    });

    const selected = selectPersonaMemories({
      candidates: [...episodes, style, belief],
      context: turnPlan.context,
    });

    const kinds = selected.map((candidate) => candidate.kind);
    expect(kinds).toContain("style");
    expect(kinds).toContain("belief");
    expect(kinds.filter((kind) => kind === "episode")).toHaveLength(3);
  });

  it("caps every memory kind at its budget", () => {
    const beliefs = Array.from({ length: 8 }, (_, index) =>
      buildCandidate({ id: `belief_${index}`, kind: "belief" })
    );

    const selected = selectPersonaMemories({
      candidates: beliefs,
      context: turnPlan.context,
    });

    expect(selected).toHaveLength(4);
  });

  it("ranks embedding similarity above lexically blind candidates", () => {
    // Korean memory text shares no tokens with the English retrieval query;
    // only the stored embedding similarity can surface it.
    const semanticMatch = buildCandidate({
      id: "episode_semantic",
      kind: "episode",
      semanticSimilarity: 0.88,
      text: "단순함을 지키려고 기능 열두 개를 잘라냈다",
    });
    const unrelated = Array.from({ length: 4 }, (_, index) =>
      buildCandidate({
        id: `episode_unrelated_${index}`,
        kind: "episode",
        semanticSimilarity: 0.56,
        text: "관련 없는 다른 이야기",
      })
    );

    const selected = selectPersonaMemories({
      candidates: [...unrelated, semanticMatch],
      context: turnPlan.context,
    });

    const episodeIds = selected
      .filter((candidate) => candidate.kind === "episode")
      .map((candidate) => candidate.id);
    expect(episodeIds[0]).toBe("episode_semantic");
  });

  it("spreads activation from a strong belief to its supporting episode", () => {
    // The strongly held belief links to an episode that shares no tokens with
    // the query; association should pull the episode ahead of its siblings.
    const belief = buildCandidate({
      id: "belief_1",
      kind: "belief",
      linkIds: ["episode_supporting"],
      strength: 0.95,
    });
    const supported = buildCandidate({
      id: "episode_supporting",
      kind: "episode",
      text: "전혀 관련 없는 텍스트",
    });
    const siblings = Array.from({ length: 5 }, (_, index) =>
      buildCandidate({
        id: `episode_sibling_${index}`,
        kind: "episode",
        text: "전혀 관련 없는 텍스트",
      })
    );

    const selected = selectPersonaMemories({
      candidates: [belief, ...siblings, supported],
      context: turnPlan.context,
    });

    const episodeIds = selected
      .filter((candidate) => candidate.kind === "episode")
      .map((candidate) => candidate.id);
    expect(episodeIds[0]).toBe("episode_supporting");
  });

  it("resolves links through alias ids such as parent document ids", () => {
    const episode = buildCandidate({
      id: "episode_1",
      kind: "episode",
      linkIds: ["document_1"],
      text: "product launch simplicity keynote",
    });
    const linkedChunk = buildCandidate({
      aliasIds: ["document_1"],
      id: "chunk_linked",
      kind: "source",
      text: "전혀 관련 없는 텍스트",
    });
    const otherChunks = Array.from({ length: 4 }, (_, index) =>
      buildCandidate({
        id: `chunk_other_${index}`,
        kind: "source",
        text: "전혀 관련 없는 텍스트",
      })
    );

    const selected = selectPersonaMemories({
      candidates: [episode, ...otherChunks, linkedChunk],
      context: turnPlan.context,
    });

    const sourceIds = selected
      .filter((candidate) => candidate.kind === "source")
      .map((candidate) => candidate.id);
    expect(sourceIds[0]).toBe("chunk_linked");
  });

  it("matches Korean item questions to source chunks despite object particles", () => {
    const itemTurnPlan: PersonaTurnPlan = {
      ...turnPlan,
      context: {
        ...turnPlan.context,
        contextKeys: {
          ...turnPlan.context.contextKeys,
          entities: ["칫솔"],
          themes: ["개인 물건"],
        },
        retrievalQueries: {
          ...turnPlan.context.retrievalQueries,
          source: "칫솔 브랜드 사용 제품",
        },
      },
    };
    const matching = buildCandidate({
      id: "chunk_toothbrush",
      kind: "source",
      text: "영상에서 한서린은 칫솔을 고를 때 세라딘 브랜드를 쓴다고 말한다.",
    });
    const unrelated = Array.from({ length: 4 }, (_, index) =>
      buildCandidate({
        id: `chunk_other_${index}`,
        kind: "source",
        text: "브랜드 전략과 메뉴 구성에 대한 다른 이야기",
      })
    );

    const selected = selectPersonaMemories({
      candidates: [...unrelated.slice(0, 1), matching, ...unrelated.slice(1)],
      context: itemTurnPlan.context,
    });

    const sourceIds = selected
      .filter((candidate) => candidate.kind === "source")
      .map((candidate) => candidate.id);
    expect(sourceIds[0]).toBe("chunk_toothbrush");
  });

  it("matches Korean item questions to source-backed facts", () => {
    const itemTurnPlan: PersonaTurnPlan = {
      ...turnPlan,
      context: {
        ...turnPlan.context,
        contextKeys: {
          ...turnPlan.context.contextKeys,
          entities: ["칫솔", "세라딘"],
          themes: ["개인 물건"],
        },
        retrievalQueries: {
          ...turnPlan.context.retrievalQueries,
          source: "칫솔 브랜드 세라딘 사용 제품",
        },
      },
    };
    const matching = buildCandidate({
      confidence: 0.92,
      id: "fact_toothbrush",
      kind: "fact",
      summary: "uses_product: 한서린은 source 영상에서 세라딘 칫솔을 사용한다.",
      text: "uses_product product 세라딘 칫솔 seradin toothbrush 한서린은 source 영상에서 세라딘 칫솔을 사용한다",
      themes: ["uses_product", "product", "세라딘 칫솔"],
      voice: "저는 세라딘 칫솔을 써요.",
    });
    const unrelated = buildCandidate({
      id: "fact_other",
      kind: "fact",
      text: "located_at place 레스토랑 매장 운영",
    });

    const result = selectPersonaMemoriesWithScores({
      candidates: [unrelated, matching],
      context: itemTurnPlan.context,
    });

    expect(result.selected.map((candidate) => candidate.id)).toContain(
      "fact_toothbrush"
    );
    expect(result.retrieval.budgets.fact).toBe(4);
    const score = result.retrieval.candidates.find(
      (candidate) => candidate.id === "fact_toothbrush"
    );
    expect(score).toMatchObject({
      kind: "fact",
      query: itemTurnPlan.context.retrievalQueries.source,
      selected: true,
    });
  });

  it("filters out sensitive memory", () => {
    const selected = selectPersonaMemories({
      candidates: [
        buildCandidate({
          id: "episode_sensitive",
          kind: "episode",
          privacyLevel: "sensitive",
        }),
      ],
      context: turnPlan.context,
    });

    expect(selected).toHaveLength(0);
  });

  it("records retrieval score diagnostics for later inspection", () => {
    const semanticMatch = buildCandidate({
      id: "belief_semantic",
      kind: "belief",
      semanticSimilarity: 0.9,
      strength: 0.9,
      text: "브랜드 본질과 고객 경험",
      themes: ["simplicity"],
    });
    const sensitive = buildCandidate({
      id: "belief_sensitive",
      kind: "belief",
      privacyLevel: "sensitive",
      text: "simplicity",
    });

    const result = selectPersonaMemoriesWithScores({
      candidates: [semanticMatch, sensitive],
      context: turnPlan.context,
      now: new Date("2026-06-01T00:00:00.000Z"),
    });

    expect(result.selected.map((candidate) => candidate.id)).toContain(
      "belief_semantic"
    );
    expect(result.retrieval).toMatchObject({
      candidateCount: 2,
      selectedCount: 1,
      version: 1,
    });
    const semanticScore = result.retrieval.candidates.find(
      (candidate) => candidate.id === "belief_semantic"
    );
    expect(semanticScore).toMatchObject({
      activationScore: expect.any(Number),
      availability: 1,
      baseRank: expect.any(Number),
      excludedReason: null,
      query: turnPlan.context.retrievalQueries.semantic,
      selected: true,
      selectedOrder: 1,
    });
    expect(semanticScore?.components).toMatchObject({
      normalizedSemanticSimilarity: 1,
      semanticSimilarity: 1,
    });
    const sensitiveScore = result.retrieval.candidates.find(
      (candidate) => candidate.id === "belief_sensitive"
    );
    expect(sensitiveScore).toMatchObject({
      activationScore: null,
      components: null,
      excludedReason: "privacy",
      rank: 0,
      selected: false,
    });
  });
});

describe("calculateMemoryAvailability", () => {
  const now = new Date("2026-06-01T00:00:00.000Z");

  it("keeps identity memory fully available regardless of age", () => {
    const belief = buildCandidate({
      createdAt: new Date("2020-01-01T00:00:00.000Z"),
      id: "belief_old",
      kind: "belief",
    });
    expect(calculateMemoryAvailability({ candidate: belief, now })).toBe(1);
  });

  it("fades old episodes but never below the floor", () => {
    const fresh = buildCandidate({
      createdAt: new Date("2026-05-30T00:00:00.000Z"),
      id: "episode_fresh",
      kind: "episode",
    });
    const stale = buildCandidate({
      createdAt: new Date("2024-01-01T00:00:00.000Z"),
      id: "episode_stale",
      kind: "episode",
    });
    const freshAvailability = calculateMemoryAvailability({
      candidate: fresh,
      now,
    });
    const staleAvailability = calculateMemoryAvailability({
      candidate: stale,
      now,
    });
    expect(freshAvailability).toBeGreaterThan(staleAvailability);
    expect(staleAvailability).toBeGreaterThanOrEqual(0.35);
  });

  it("retains emotionally salient episodes longer", () => {
    const calm = buildCandidate({
      createdAt: new Date("2025-09-01T00:00:00.000Z"),
      id: "episode_calm",
      kind: "episode",
      sourceConfidence: 0.1,
    });
    const intense = buildCandidate({
      createdAt: new Date("2025-09-01T00:00:00.000Z"),
      id: "episode_intense",
      kind: "episode",
      sourceConfidence: 0.95,
    });
    expect(
      calculateMemoryAvailability({ candidate: intense, now })
    ).toBeGreaterThan(calculateMemoryAvailability({ candidate: calm, now }));
  });

  it("primes recently recalled and frequently used memories", () => {
    const dormant = buildCandidate({
      createdAt: new Date("2024-06-01T00:00:00.000Z"),
      id: "episode_dormant",
      kind: "episode",
    });
    const rehearsed = buildCandidate({
      activationCount: 12,
      createdAt: new Date("2024-06-01T00:00:00.000Z"),
      id: "episode_rehearsed",
      kind: "episode",
      lastActivatedAt: new Date("2026-05-29T00:00:00.000Z"),
    });
    expect(
      calculateMemoryAvailability({ candidate: rehearsed, now })
    ).toBeGreaterThan(calculateMemoryAvailability({ candidate: dormant, now }));
  });
});

describe("selectPersonaMemories with forgetting", () => {
  it("prefers a rehearsed memory over a dormant equal sibling", () => {
    const now = new Date("2026-06-01T00:00:00.000Z");
    const shared = {
      createdAt: new Date("2024-06-01T00:00:00.000Z"),
      kind: "episode" as const,
      text: "product launch simplicity keynote",
    };
    const dormant = buildCandidate({ ...shared, id: "episode_dormant" });
    const rehearsed = buildCandidate({
      ...shared,
      activationCount: 12,
      id: "episode_rehearsed",
      lastActivatedAt: new Date("2026-05-29T00:00:00.000Z"),
    });

    const selected = selectPersonaMemories({
      candidates: [dormant, rehearsed],
      context: turnPlan.context,
      now,
    });

    const episodeIds = selected
      .filter((candidate) => candidate.kind === "episode")
      .map((candidate) => candidate.id);
    expect(episodeIds[0]).toBe("episode_rehearsed");
  });
});

describe("mood-congruent recall", () => {
  it("prefers episodes whose valence matches the current mood", () => {
    const shared = {
      kind: "episode" as const,
      text: "product launch simplicity keynote",
    };
    const sad = buildCandidate({
      ...shared,
      affect: { arousal: 0.5, dominance: 0.4, valence: -0.8 },
      id: "episode_sad",
    });
    const happy = buildCandidate({
      ...shared,
      affect: { arousal: 0.5, dominance: 0.4, valence: 0.8 },
      id: "episode_happy",
    });

    const selected = selectPersonaMemories({
      candidates: [happy, sad],
      context: turnPlan.context,
      mood: { arousal: 0.4, dominance: 0.4, valence: -0.7 },
    });

    const episodeIds = selected
      .filter((candidate) => candidate.kind === "episode")
      .map((candidate) => candidate.id);
    expect(episodeIds[0]).toBe("episode_sad");
  });

  it("uses arousal and dominance as secondary mood congruence signals", () => {
    const shared = {
      affect: { arousal: 0.2, dominance: 0.9, valence: -0.7 },
      kind: "episode" as const,
      text: "product launch simplicity keynote",
    };
    const sameValenceWrongEnergy = buildCandidate({
      ...shared,
      id: "episode_flat",
    });
    const padMatch = buildCandidate({
      ...shared,
      affect: { arousal: 0.9, dominance: 0.2, valence: -0.7 },
      id: "episode_tense",
    });

    const result = selectPersonaMemoriesWithScores({
      candidates: [sameValenceWrongEnergy, padMatch],
      context: turnPlan.context,
      mood: { arousal: 0.9, dominance: 0.2, valence: -0.7 },
    });

    const episodeIds = result.selected
      .filter((candidate) => candidate.kind === "episode")
      .map((candidate) => candidate.id);
    expect(episodeIds[0]).toBe("episode_tense");
    const tenseScore = result.retrieval.candidates.find(
      (candidate) => candidate.id === "episode_tense"
    );
    const flatScore = result.retrieval.candidates.find(
      (candidate) => candidate.id === "episode_flat"
    );
    expect(tenseScore?.components?.moodCongruence).toBeGreaterThan(
      flatScore?.components?.moodCongruence ?? 1
    );
  });

  it("adds a small bounded bonus for matching emotion labels", () => {
    const shared = {
      affect: { arousal: 0.7, dominance: 0.3, valence: -0.45 },
      kind: "episode" as const,
      text: "product launch simplicity keynote",
    };
    const differentLabel = buildCandidate({
      ...shared,
      emotions: [{ emotion: "joy", intensity: 0.9 }],
      id: "episode_joy",
    });
    const matchingLabel = buildCandidate({
      ...shared,
      emotions: [{ emotion: "anxiety", intensity: 0.9 }],
      id: "episode_anxiety",
    });

    const result = selectPersonaMemoriesWithScores({
      candidates: [differentLabel, matchingLabel],
      context: turnPlan.context,
      mood: { arousal: 0.7, dominance: 0.3, valence: -0.45 },
      moodEmotionLabels: [
        { intensity: 1, label: "anxiety", source: "context_emotion" },
      ],
    });

    const episodeIds = result.selected
      .filter((candidate) => candidate.kind === "episode")
      .map((candidate) => candidate.id);
    expect(episodeIds[0]).toBe("episode_anxiety");
    const matchingScore = result.retrieval.candidates.find(
      (candidate) => candidate.id === "episode_anxiety"
    );
    const differentScore = result.retrieval.candidates.find(
      (candidate) => candidate.id === "episode_joy"
    );
    expect(matchingScore?.components?.emotionLabelOverlap).toBeGreaterThan(0);
    expect(differentScore?.components?.emotionLabelOverlap).toBe(0);
    expect(matchingScore?.components?.affectiveSalience).toBeGreaterThan(
      differentScore?.components?.affectiveSalience ?? 1
    );
  });
});

describe("normalizeCosineSimilarity", () => {
  it("maps the unrelated-text baseline to zero and strong matches near one", () => {
    expect(normalizeCosineSimilarity(null)).toBe(0);
    expect(normalizeCosineSimilarity(0.55)).toBe(0);
    expect(normalizeCosineSimilarity(0.9)).toBe(1);
    expect(normalizeCosineSimilarity(0.725)).toBeCloseTo(0.5, 2);
    expect(normalizeCosineSimilarity(Number.NaN)).toBe(0);
  });
});
