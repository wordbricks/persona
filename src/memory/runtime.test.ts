import { describe, expect, it } from "vitest";

import type { PersonaProfile, PersonaWorkspacePayload } from "../schema";
import { calculatePersonaMoodUpdate } from "./mood";
import {
  evaluatePostResponseInteractionMemory,
  formatWorkspacePrompt,
  planPersonaTurnWithLlm,
  shouldRecordInteractionMemory,
  triageInteractionMemoryWithLlm,
  triagePostResponseInteractionMemoryWithLlm,
} from "./runtime";
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

const profile: PersonaProfile = {
  consentStatus: "fictional_or_authorized",
  createdAt: new Date("2026-01-01T00:00:00.000Z"),
  displayName: "Juno",
  id: "persona_1",
  organizationId: "org_1",
  personaKey: "juno",
  personaScope: "organization",
  personaType: "simulated_character",
  personaVersion: "v1",
  policy: {
    allowedUse: ["private_chat"],
    biographicalSummary: "Product visionary persona.",
    forbiddenUse: ["deceptive_impersonation"],
    knowledgeCutoffForPersona: "2024-01-01",
    transparencyLabel: "AI persona simulation for Juno.",
  },
  profile: {},
  sourceRef: null,
  state: "active",
  updatedAt: new Date("2026-01-01T00:00:00.000Z"),
  updatedByUserId: null,
};

const workspacePayload: PersonaWorkspacePayload = {
  activeBeliefs: [],
  activeFacts: [],
  activeHabits: [],
  activeMemories: [],
  activeStyleProfiles: [],
  affectiveState: {},
  appraisalSummary: { averageConfidence: 0.8 },
  contextKeys: {},
  gateOutput: {},
  responsePlan: {},
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

describe("persona turn planner prompt", () => {
  it("accepts planner turns that do not need pre-activated memory", async () => {
    await expect(
      planPersonaTurnWithLlm({
        llm: async () => ({
          ...turnPlan,
          gate: {
            ...turnPlan.gate,
            neededMemoryTypes: [],
          },
        }),
        message: "Run a safe shell smoke test.",
        persona: profile,
      })
    ).resolves.toMatchObject({
      gate: {
        neededMemoryTypes: [],
      },
    });
  });

  it("normalizes planner time arrays into the queryFocus time value", async () => {
    await expect(
      planPersonaTurnWithLlm({
        llm: async () => ({
          ...turnPlan,
          gate: {
            ...turnPlan.gate,
            queryFocus: {
              ...turnPlan.gate.queryFocus,
              time: ["last week", "this morning"],
            },
          },
        }),
        message: "What did I care about last week and this morning?",
        persona: profile,
      })
    ).resolves.toMatchObject({
      gate: {
        queryFocus: {
          time: "last week, this morning",
        },
      },
    });
  });

  it("normalizes empty planner time arrays to null", async () => {
    await expect(
      planPersonaTurnWithLlm({
        llm: async () => ({
          ...turnPlan,
          gate: {
            ...turnPlan.gate,
            queryFocus: {
              ...turnPlan.gate.queryFocus,
              time: [],
            },
          },
        }),
        message: "What should I do?",
        persona: profile,
      })
    ).resolves.toMatchObject({
      gate: {
        queryFocus: {
          time: null,
        },
      },
    });
  });
});

describe("shouldRecordInteractionMemory", () => {
  it("triggers on Korean memory requests", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "이건 다음에도 꼭 기억해줘",
        turnPlan,
      })
    ).toBe(true);
  });

  it("triggers on English memory requests", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "Please remember this preference.",
        turnPlan,
      })
    ).toBe(true);
  });

  it("triggers on Korean polite negation forms", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "그건 기억하지 말아줘",
        turnPlan,
      })
    ).toBe(true);
    expect(
      shouldRecordInteractionMemory({
        message: "잊지 말아줘",
        turnPlan,
      })
    ).toBe(true);
  });

  it("triggers on English requests with a curly apostrophe", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "Don’t forget this.",
        turnPlan,
      })
    ).toBe(true);
  });

  it("does not trigger on plain questions", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "단순함에 대해 어떻게 생각해?",
        turnPlan,
      })
    ).toBe(false);
  });

  it("does not trigger on Korean recall questions", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "내 생일 기억해?",
        turnPlan,
      })
    ).toBe(false);
  });

  it("triggers on emotional reflection turns", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "요즘 회사 일 때문에 너무 지쳤어",
        turnPlan: {
          ...turnPlan,
          gate: { ...turnPlan.gate, queryType: "emotional_reflection" },
        },
      })
    ).toBe(true);
  });

  it("records novel autobiographical turns when embeddings show no coverage", () => {
    const novelTurnPlan: PersonaTurnPlan = {
      ...turnPlan,
      gate: { ...turnPlan.gate, queryType: "autobiographical_fact" },
    };
    const lowCoverage = [
      buildCandidate({
        id: "episode_1",
        kind: "episode",
        semanticSimilarity: 0.57,
      }),
    ];
    const highCoverage = [
      buildCandidate({
        id: "episode_1",
        kind: "episode",
        semanticSimilarity: 0.85,
      }),
    ];

    expect(
      shouldRecordInteractionMemory({
        message: "처음 듣는 이야기인데 한번 들어봐",
        selected: lowCoverage,
        turnPlan: novelTurnPlan,
      })
    ).toBe(true);
    expect(
      shouldRecordInteractionMemory({
        message: "그때 얘기 또 해줄게",
        selected: highCoverage,
        turnPlan: novelTurnPlan,
      })
    ).toBe(false);
  });

  it("stays conservative without embedding similarities", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "새로운 이야기",
        selected: [
          buildCandidate({
            id: "episode_1",
            kind: "episode",
            semanticSimilarity: null,
          }),
        ],
        turnPlan: {
          ...turnPlan,
          gate: { ...turnPlan.gate, queryType: "autobiographical_fact" },
        },
      })
    ).toBe(false);
  });

  it("triggers on relationship questions", () => {
    expect(
      shouldRecordInteractionMemory({
        message: "우리 지난번에 무슨 얘기했지?",
        turnPlan: {
          ...turnPlan,
          gate: { ...turnPlan.gate, queryType: "relationship_question" },
        },
      })
    ).toBe(true);
  });
});

describe("triageInteractionMemoryWithLlm", () => {
  it("records implicit durable preferences that hard rules miss", async () => {
    const decision = await triageInteractionMemoryWithLlm({
      llm: async () => ({
        confidence: 0.84,
        dedupeKey: "dislikes-large-prs",
        memoryIntent: "user_preference",
        memoryType: "preference",
        privacyLevel: "private",
        reason: "The user expressed a stable review preference.",
        shouldRemember: true,
        summary:
          "The user dislikes reviewing large PRs and prefers smaller reviewable changes.",
        themes: ["code_review"],
      }),
      message: "긴 PR 보면 리뷰하기 싫더라.",
      persona: profile,
      turnPlan,
    });

    expect(
      shouldRecordInteractionMemory({
        message: "긴 PR 보면 리뷰하기 싫더라.",
        turnPlan,
      })
    ).toBe(false);
    expect(decision).toMatchObject({
      memoryIntent: "user_preference",
      memoryType: "preference",
      shouldRemember: true,
      source: "llm",
    });
  });

  it("lets explicit memory requests survive a conservative LLM false negative", async () => {
    const decision = await triageInteractionMemoryWithLlm({
      llm: async () => ({
        confidence: 0.4,
        dedupeKey: null,
        memoryIntent: "none",
        memoryType: "none",
        privacyLevel: "private",
        reason: "Too vague.",
        shouldRemember: false,
        summary: null,
        themes: [],
      }),
      message: "나는 작은 PR을 선호한다고 기억해줘",
      persona: profile,
      turnPlan,
    });

    expect(decision).toMatchObject({
      memoryIntent: "explicit_memory_request",
      shouldRemember: true,
      source: "hard_rule",
    });
  });

  it("blocks sensitive material even when explicitly requested", async () => {
    const decision = await triageInteractionMemoryWithLlm({
      llm: async () => ({
        confidence: 0.95,
        dedupeKey: "password",
        memoryIntent: "explicit_memory_request",
        memoryType: "interaction",
        privacyLevel: "private",
        reason: "The user asked to remember it.",
        shouldRemember: true,
        summary: "The user shared a password.",
        themes: ["security"],
      }),
      message: "내 비밀번호는 hunter2라고 기억해줘",
      persona: profile,
      turnPlan,
    });

    expect(decision).toMatchObject({
      privacyLevel: "sensitive",
      shouldRemember: false,
    });
  });

  it("falls back to hard rules when LLM triage fails", async () => {
    const decision = await triageInteractionMemoryWithLlm({
      llm: async () => {
        throw new Error("llm unavailable");
      },
      message: "요즘 회사 일 때문에 너무 지쳤어",
      persona: profile,
      turnPlan: {
        ...turnPlan,
        gate: { ...turnPlan.gate, queryType: "emotional_reflection" },
      },
    });

    expect(decision).toMatchObject({
      memoryIntent: "emotional_salience",
      shouldRemember: true,
      source: "llm_fallback",
    });
  });
});

describe("triagePostResponseInteractionMemoryWithLlm", () => {
  it("uses LLM triage for durable post-response lessons", async () => {
    const decision = await triagePostResponseInteractionMemoryWithLlm({
      assistantMessage:
        "다음엔 KPI 목표를 필요한 생성 수, 가능한 모수, 도달률, 전환율로 먼저 역산하겠습니다.",
      llm: async () => ({
        confidence: 0.88,
        dedupeKey: "kpi-backsolve-first",
        memoryIntent: "durable_product_lesson",
        memoryType: "lesson",
        privacyLevel: "private",
        reason: "The assistant committed to a reusable product analysis rule.",
        shouldRemember: true,
        summary:
          "When evaluating KPI goals, first backsolve required volume, reachable audience, reach rate, and conversion.",
        themes: ["kpi", "product"],
      }),
      persona: profile,
      turnPlan: {
        ...turnPlan,
        gate: { ...turnPlan.gate, queryType: "value_judgment" },
      },
      userMessage: "kelly 피드백에 대해 어떻게 생각해?",
    });

    expect(decision).toMatchObject({
      memoryIntent: "durable_product_lesson",
      memoryType: "lesson",
      shouldRemember: true,
      source: "llm",
    });
  });
});

describe("evaluatePostResponseInteractionMemory", () => {
  it("records self-correction feedback after the assistant accepts it", () => {
    const evaluation = evaluatePostResponseInteractionMemory({
      assistantMessage:
        "동의해요. 특히 48명 고빈도 유저에 너무 오래 머문 건 제 판단 미스였습니다. 다음엔 KPI 목표를 필요한 생성 수, 가능한 모수, 도달률, 전환율, 유저당 생성 수로 먼저 역산하겠습니다.",
      turnPlan: {
        ...turnPlan,
        gate: { ...turnPlan.gate, queryType: "value_judgment" },
      },
      userMessage: "kelly 피드백에 대해 어떻게 생각해?",
    });

    expect(evaluation.shouldRemember).toBe(true);
    expect(evaluation.memoryIntent).toBe("self_correction_feedback");
  });

  it("skips routine value judgments without durable feedback", () => {
    const evaluation = evaluatePostResponseInteractionMemory({
      assistantMessage:
        "저는 홈 배너가 모달보다 낫다고 봅니다. 도달률이 높고 방해감이 낮기 때문입니다.",
      turnPlan,
      userMessage: "모달이 좋아 배너가 좋아?",
    });

    expect(evaluation.shouldRemember).toBe(false);
    expect(evaluation.memoryIntent).toBeNull();
  });
});

describe("formatWorkspacePrompt", () => {
  it("surfaces first-person memory voice and knowledge cutoff", () => {
    const prompt = formatWorkspacePrompt({
      disclosurePolicy: "on_request",
      profile,
      selected: [
        buildCandidate({
          id: "episode_1",
          kind: "episode",
          summary: "Launch keynote: cut features to ship one clear story.",
          voice: "I remember killing twelve features the week before launch.",
        }),
      ],
      turnPlan,
      workspacePayload,
    });

    expect(prompt).toContain("### Episodic memories");
    expect(prompt).toContain(
      "In the persona's own words: I remember killing twelve features"
    );
    expect(prompt).toContain("knowledgeCutoffForPersona: 2024-01-01");
    expect(prompt).toContain("Treat events after 2024-01-01");
    expect(prompt).toContain("available web or current-source tools");
    expect(prompt).toContain("persona-grounded inference");
    expect(prompt).toContain(
      "Activated Source-backed facts and Source excerpts count as source context"
    );
    expect(prompt).toContain("concrete product, brand");
    expect(prompt).toContain("말하면 안 됩니다");
    expect(prompt).not.toContain("transparencyLabel");
    expect(prompt).not.toContain("AI persona simulation for Juno.");
  });

  it("only exposes the transparency label for transparent identity answers", () => {
    const prompt = formatWorkspacePrompt({
      disclosurePolicy: "on_request",
      profile,
      selected: [],
      turnPlan: {
        ...turnPlan,
        gate: {
          ...turnPlan.gate,
          answerMode: "transparent_meta_response",
          queryType: "identity_confusion",
          safetyFlags: ["identity_confusion"],
        },
      },
      workspacePayload,
    });

    expect(prompt).toContain(
      "transparencyLabel: AI persona simulation for Juno."
    );
  });

  it("exposes the transparency label and active disclosure constraints by default", () => {
    const prompt = formatWorkspacePrompt({
      profile,
      selected: [],
      turnPlan,
      workspacePayload,
    });

    expect(prompt).toContain(
      "transparencyLabel: AI persona simulation for Juno."
    );
    expect(prompt).toContain("brief AI/persona-simulation disclosure");
    expect(prompt).toContain("AI/persona-simulation boundary visible");
  });

  it("renders empty pre-activated memory routing as none", () => {
    const prompt = formatWorkspacePrompt({
      disclosurePolicy: "on_request",
      profile,
      selected: [],
      turnPlan: {
        ...turnPlan,
        gate: {
          ...turnPlan.gate,
          neededMemoryTypes: [],
        },
      },
      workspacePayload,
    });

    expect(prompt).toContain("- neededMemoryTypes: none");
  });

  it("renders external long-term memory without provider branding", () => {
    const prompt = formatWorkspacePrompt({
      disclosurePolicy: "on_request",
      profile,
      selected: [],
      turnPlan,
      workspacePayload: {
        ...workspacePayload,
        externalMemory: {
          candidates: [],
          enabled: true,
          provider: "hindsight",
          recallAttempted: true,
          selected: [
            {
              confidence: 0.82,
              createdAt: "2026-06-24T00:00:00.000Z",
              id: "hindsight:bank:memory_1",
              kind: "external_observation",
              privacyLevel: "private",
              provenance: {
                bankId: "persona_user_org_persona_user",
                citations: ["conversation memory from 2026-06-24"],
                memoryId: "memory_1",
                provider: "hindsight",
              },
              score: {
                finalScore: 0.82,
                providerScore: 1,
              },
              sourceConfidence: 0.78,
              text: "The user prefers small PRs with clear test output.",
              themes: ["testing"],
              title: "External observation",
              updatedAt: "2026-06-24T00:00:00.000Z",
            },
          ],
          version: 1,
        },
      },
    });

    expect(prompt).toContain("### External long-term memory");
    expect(prompt).toContain(
      "The user prefers small PRs with clear test output."
    );
    expect(prompt).toContain("Do not mention the external memory provider");
    expect(prompt).not.toContain("[hindsight, confidence");
  });

  it("renders the current mood with qualitative labels", () => {
    const moodUpdate = calculatePersonaMoodUpdate({
      current: { arousal: 0.3, dominance: 0.5, valence: 0.1 },
      elapsedMs: 0,
      impulse: { arousal: 0.8, dominance: 0.2, valence: -0.5 },
      turnAffect: {
        emotionLabels: [
          { intensity: 0.8, label: "anxiety", source: "context_emotion" },
        ],
        impulse: { arousal: 0.8, dominance: 0.2, valence: -0.5 },
        reasons: ["Context emotion label anxiety was activated."],
        sources: [],
      },
    });
    const prompt = formatWorkspacePrompt({
      disclosurePolicy: "on_request",
      mood: { arousal: 0.8, dominance: 0.2, valence: -0.5 },
      moodUpdate: moodUpdate.trace,
      profile,
      selected: [],
      turnPlan,
      workspacePayload,
    });

    expect(prompt).toContain("## Persona Current Mood");
    expect(prompt).toContain("valence: -0.50 (negative)");
    expect(prompt).toContain("arousal: 0.80 (energized)");
    expect(prompt).toContain("dominance: 0.20 (tentative)");
    expect(prompt).toContain("emotionLabels: anxiety=0.80");
    expect(prompt).toContain("why:");
    expect(prompt).toContain("must never change facts");
  });

  it("trims overflowing memory without dropping response constraints", () => {
    const selected = Array.from({ length: 40 }, (_, index) =>
      buildCandidate({
        id: `episode_${index}`,
        kind: "episode",
        summary: "x".repeat(600),
      })
    );

    const prompt = formatWorkspacePrompt({
      disclosurePolicy: "on_request",
      profile,
      selected,
      turnPlan,
      workspacePayload,
    });

    expect(prompt).toContain("(memory context truncated)");
    expect(prompt).toContain("Embody the persona.");
    expect(prompt).toContain("Do not open or preface ordinary responses");
    expect(prompt).toContain("## Response Constraints");
    expect(prompt.length).toBeLessThanOrEqual(12_000);
    // Whole lines are dropped, so the last memory line is never a half entry.
    const memorySection = prompt
      .split("## Active Persona Memory")[1]
      ?.split("## Structured Appraisal Summary")[0];
    expect(memorySection).toBeDefined();
    for (const line of (memorySection ?? "").split("\n")) {
      expect(
        line === "" ||
          line.startsWith("- ") ||
          line.startsWith("  - ") ||
          line.startsWith("###")
      ).toBe(true);
    }
  });
});
