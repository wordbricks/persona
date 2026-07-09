import { describe, expect, it } from "vitest";

import {
  calculatePersonaMoodUpdate,
  estimateTurnAffect,
  explainTurnAffect,
  PERSONA_MOOD_BASELINE,
  updatePersonaMood,
} from "./mood";
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

describe("updatePersonaMood", () => {
  it("decays toward baseline during silence", () => {
    const excited = { arousal: 0.9, dominance: 0.8, valence: 0.9 };
    const afterTwoDays = updatePersonaMood({
      current: excited,
      elapsedMs: 48 * 3_600_000,
      impulse: null,
    });
    expect(afterTwoDays.valence).toBeLessThan(excited.valence);
    expect(afterTwoDays.valence).toBeGreaterThan(PERSONA_MOOD_BASELINE.valence);
    expect(afterTwoDays.arousal).toBeLessThan(excited.arousal);
  });

  it("moves with an impulse but keeps inertia", () => {
    const calm = { ...PERSONA_MOOD_BASELINE };
    const negativeImpulse = { arousal: 0.8, dominance: 0.3, valence: -0.9 };
    const next = updatePersonaMood({
      current: calm,
      elapsedMs: 0,
      impulse: negativeImpulse,
    });
    expect(next.valence).toBeLessThan(calm.valence);
    expect(next.valence).toBeGreaterThan(negativeImpulse.valence);
    expect(next.arousal).toBeGreaterThan(calm.arousal);
  });

  it("starts from baseline when no mood is stored", () => {
    const next = updatePersonaMood({
      current: null,
      elapsedMs: 0,
      impulse: null,
    });
    expect(next).toEqual(PERSONA_MOOD_BASELINE);
  });

  it("returns an auditable mood update trace", () => {
    const turnAffect = explainTurnAffect({
      selected: [
        buildCandidate({
          affect: { arousal: 0.8, dominance: 0.35, valence: -0.7 },
          emotions: [{ emotion: "grief", intensity: 0.9 }],
          id: "episode_grief",
          kind: "episode",
          sourceConfidence: 0.8,
          summary: "Loss shaped the launch week.",
        }),
      ],
      turnPlan,
    });
    const result = calculatePersonaMoodUpdate({
      current: { arousal: 0.4, dominance: 0.5, valence: 0.2 },
      elapsedMs: 3_600_000,
      impulse: turnAffect.impulse,
      turnAffect,
    });

    expect(result.trace).toMatchObject({
      decayHalfLifeHours: 24,
      elapsedMs: 3_600_000,
      impulse: expect.any(Object),
      previousMood: { arousal: 0.4, dominance: 0.5, valence: 0.2 },
      result: result.mood,
      version: 1,
    });
    expect(result.trace.emotionLabels).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "grief",
          memoryId: "episode_grief",
          source: "activated_memory",
        }),
      ])
    );
    expect(result.trace.reasons.join(" ")).toContain(
      "Averaged 1 activated memory affect annotation"
    );
  });
});

describe("estimateTurnAffect", () => {
  it("averages activated episode affect", () => {
    const affectCandidates = [
      buildCandidate({
        affect: { arousal: 0.8, dominance: 0.4, valence: -0.6 },
        id: "episode_sad",
        kind: "episode",
      }),
      buildCandidate({
        affect: { arousal: 0.4, dominance: 0.6, valence: -0.2 },
        id: "episode_low",
        kind: "episode",
      }),
    ];
    const impulse = estimateTurnAffect({
      selected: affectCandidates,
      turnPlan,
    });
    expect(impulse?.valence).toBeCloseTo(-0.4, 5);
  });

  it("returns null when the turn carries no affect signal", () => {
    expect(
      estimateTurnAffect({
        selected: [buildCandidate({ id: "episode_1", kind: "episode" })],
        turnPlan,
      })
    ).toBeNull();
  });

  it("uses query cues for emotional turns without annotated memories", () => {
    const impulse = estimateTurnAffect({
      selected: [],
      turnPlan: {
        ...turnPlan,
        gate: { ...turnPlan.gate, queryType: "emotional_reflection" },
      },
    });
    expect(impulse?.arousal).toBeGreaterThan(0.5);
  });

  it("uses context emotions as an immediate turn impulse", () => {
    const estimate = explainTurnAffect({
      selected: [],
      turnPlan: {
        ...turnPlan,
        context: {
          ...turnPlan.context,
          contextKeys: {
            ...turnPlan.context.contextKeys,
            emotions: ["anxiety"],
          },
        },
      },
    });

    expect(estimate.impulse).not.toBeNull();
    if (!estimate.impulse) {
      throw new Error("Expected context emotion to produce an impulse.");
    }
    expect(estimate.impulse.valence).toBeLessThan(0);
    expect(estimate.reasons.join(" ")).toContain("context emotion label");
  });

  it("adds emotion labels and source reasons to the turn affect estimate", () => {
    const estimate = explainTurnAffect({
      selected: [
        buildCandidate({
          affect: { arousal: 0.6, dominance: 0.4, valence: -0.4 },
          emotions: [
            {
              emotion: "sadness",
              evidence: "The memory is annotated as sadness.",
              intensity: 0.75,
            },
          ],
          id: "episode_sadness",
          kind: "episode",
        }),
      ],
      turnPlan: {
        ...turnPlan,
        context: {
          ...turnPlan.context,
          contextKeys: {
            ...turnPlan.context.contextKeys,
            emotions: ["regret"],
          },
        },
      },
    });

    expect(estimate.emotionLabels.map((label) => label.label)).toEqual(
      expect.arrayContaining(["sadness", "regret"])
    );
    expect(estimate.sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "episode_sadness",
          kind: "activated_memory",
        }),
      ])
    );
    expect(estimate.reasons.join(" ")).toContain(
      "activated memory affect annotation"
    );
  });
});
