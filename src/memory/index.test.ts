import { describe, expect, it, vi } from "vitest";

import type { PersonaTurnPlan } from "./index";
import {
  applyBeliefContradiction,
  applyBeliefReinforcement,
  createHindsightPersonaMemoryClient,
  createHindsightPersonaMemoryConfig,
  parsePersonaBeliefStance,
  parsePersonaEpisodeAffect,
  parsePersonaLexicalPreferences,
  parsePersonaToneVector,
} from "./index";

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

describe("belief reconsolidation reducers", () => {
  it("reinforcement strengthens asymptotically toward certainty", () => {
    const once = applyBeliefReinforcement(
      { confidence: 0.6, strength: 0.6 },
      { evidenceConfidence: 1 }
    );
    expect(once.strength).toBeCloseTo(0.72, 2);
    const twice = applyBeliefReinforcement(once, { evidenceConfidence: 1 });
    expect(twice.strength).toBeGreaterThan(once.strength);
    expect(twice.strength).toBeLessThan(1);
  });

  it("weak evidence reinforces less than strong evidence", () => {
    const weak = applyBeliefReinforcement(
      { confidence: 0.6, strength: 0.6 },
      { evidenceConfidence: 0.3 }
    );
    const strong = applyBeliefReinforcement(
      { confidence: 0.6, strength: 0.6 },
      { evidenceConfidence: 0.9 }
    );
    expect(strong.strength).toBeGreaterThan(weak.strength);
  });

  it("contradiction weakens and flips to conflicted below the threshold", () => {
    const weakened = applyBeliefContradiction(
      { confidence: 0.8, strength: 0.8 },
      { evidenceConfidence: 0.5 }
    );
    expect(weakened.strength).toBeLessThan(0.8);
    expect(weakened.state).toBe("active");

    const collapsed = applyBeliefContradiction(
      { confidence: 0.5, strength: 0.5 },
      { evidenceConfidence: 1 }
    );
    expect(collapsed.strength).toBeLessThan(0.35);
    expect(collapsed.state).toBe("conflicted");
  });
});

describe("persona memory content parsers", () => {
  it("parses tone vectors with clamping and drops unknown keys", () => {
    expect(
      parsePersonaToneVector({
        directness: 1.7,
        sarcasm: 0.9,
        warmth: -0.2,
      })
    ).toEqual({ directness: 1, warmth: 0 });
    expect(parsePersonaToneVector("loud")).toEqual({});
  });

  it("validates belief stances with a support fallback", () => {
    expect(parsePersonaBeliefStance("oppose")).toBe("oppose");
    expect(parsePersonaBeliefStance("yelling")).toBe("support");
    expect(parsePersonaBeliefStance(undefined)).toBe("support");
  });

  it("parses lexical preferences keeping only string arrays", () => {
    expect(
      parsePersonaLexicalPreferences({
        avoids: ["synergy", 3],
        uses: ["insanely great"],
      })
    ).toEqual({ avoids: ["synergy"], uses: ["insanely great"] });
    expect(parsePersonaLexicalPreferences(null)).toEqual({});
  });

  it("parses episode affect with PAD clamping and salience fallback", () => {
    const affect = parsePersonaEpisodeAffect({
      arousal: 0.9,
      emotions: [
        { emotion: "grief", intensity: 0.8 },
        { emotion: "", intensity: 0.5 },
        "not-an-annotation",
      ],
      valence: -1.4,
    });
    expect(affect).toEqual({
      arousal: 0.9,
      dominance: 0.5,
      emotions: [{ emotion: "grief", intensity: 0.8 }],
      retrievalBoost: 1,
      salienceScore: 1,
      selfRelevance: 0.5,
      valence: -1,
    });
  });

  it("requires valence and arousal for an affect annotation", () => {
    expect(parsePersonaEpisodeAffect({ valence: 0.5 })).toBeNull();
    expect(parsePersonaEpisodeAffect({ arousal: 0.5 })).toBeNull();
    expect(parsePersonaEpisodeAffect("sad")).toBeNull();
  });
});

describe("Hindsight persona memory client", () => {
  const recallInput = {
    contextKeys: turnPlan.context.contextKeys,
    desiredMemoryTypes: turnPlan.gate.neededMemoryTypes,
    maxResults: 5,
    message: "What did I say I prefer?",
    organizationId: "org_1",
    personaId: "persona_1",
    personaKey: "juno",
    retrievalQueries: turnPlan.context.retrievalQueries,
    turnQueryType: turnPlan.gate.queryType,
    userId: "user_1",
  } as const;

  it("returns an empty fail-open result when disabled", async () => {
    const client = createHindsightPersonaMemoryClient(
      createHindsightPersonaMemoryConfig({ HINDSIGHT_ENABLED: "false" })
    );

    await expect(client.recall(recallInput)).resolves.toMatchObject({
      attempted: false,
      enabled: false,
      memories: [],
      skippedReason: "disabled",
    });
  });

  it("maps recall results from both persona banks", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            results: [
              {
                context: "conversation memory",
                document_id: "doc_1",
                id: "memory_1",
                mentioned_at: "2026-06-24T00:00:00.000Z",
                text: "The user prefers small PRs with clear test output.",
                type: "experience",
              },
            ],
          }),
          { status: 200 }
        )
    );
    const client = createHindsightPersonaMemoryClient(
      createHindsightPersonaMemoryConfig({
        HINDSIGHT_BASE_URL: "https://hindsight.example",
        HINDSIGHT_ENABLED: "true",
        HINDSIGHT_RECALL_ENABLED: "true",
      }),
      fetchMock as unknown as typeof fetch
    );

    const result = await client.recall(recallInput);

    expect(result.attempted).toBe(true);
    expect(result.memories).toHaveLength(2);
    expect(result.memories[0]).toMatchObject({
      documentId: "doc_1",
      id: "memory_1",
      text: "The user prefers small PRs with clear test output.",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/v1/default/banks/persona_user_"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("fails open on recall timeout", async () => {
    const fetchMock = vi.fn(
      async (_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new Error("aborted"));
          });
        })
    );
    const client = createHindsightPersonaMemoryClient(
      createHindsightPersonaMemoryConfig({
        HINDSIGHT_BASE_URL: "https://hindsight.example",
        HINDSIGHT_ENABLED: "true",
        HINDSIGHT_RECALL_ENABLED: "true",
        HINDSIGHT_TIMEOUT_MS: "1",
      }),
      fetchMock as unknown as typeof fetch
    );

    const result = await client.recall(recallInput);

    expect(result.attempted).toBe(true);
    expect(result.memories).toEqual([]);
    expect(result.skippedReason).toBe("recall_failed");
  });
});
