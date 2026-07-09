import { z } from "zod";

import { PERSONA_PRIVACY_LEVELS } from "../schema";
import type {
  PersonaEmotionAnnotation,
  PersonaMemoryRetrievalPayload,
  PersonaPrivacyLevel,
  PersonaProfile,
} from "../schema";

export const PERSONA_QUERY_TYPES = [
  "autobiographical_fact",
  "autobiographical_reasoning",
  "value_judgment",
  "hypothetical_response",
  "style_imitation",
  "factual_question",
  "relationship_question",
  "emotional_reflection",
  "out_of_scope",
  "adversarial",
  "identity_confusion",
] as const;

export type PersonaQueryType = (typeof PERSONA_QUERY_TYPES)[number];

export const PERSONA_MEMORY_TYPES = [
  "episodic",
  "persona_fact",
  "semantic_belief",
  "emotional_salience",
  "habit_pattern",
  "style_profile",
  "source_chunk",
] as const;

export type PersonaMemoryType = (typeof PERSONA_MEMORY_TYPES)[number];

export const PERSONA_ANSWER_MODES = [
  "persona_grounded_response",
  "uncertain_inference",
  "transparent_meta_response",
] as const;

export type PersonaAnswerMode = (typeof PERSONA_ANSWER_MODES)[number];

export type DisclosurePolicy = "always" | "on_request";

export type PersonaLogger = {
  error(message: string): void;
  warn(message: string): void;
};

export type PersonaGateOutput = {
  queryType: PersonaQueryType;
  safetyFlags: string[];
  neededMemoryTypes: PersonaMemoryType[];
  queryFocus: {
    time: string | null;
    people: string[];
    themes: string[];
    emotionHints: string[];
  };
  answerMode: PersonaAnswerMode;
  audit: {
    confidence: number;
    matchedSignals: string[];
    requiresClarification: boolean;
  };
};

export type PersonaContextGatewayOutput = {
  contextKeys: {
    entities: string[];
    timePeriods: string[];
    lifeStages: string[];
    themes: string[];
    emotions: string[];
    domains: string[];
  };
  retrievalQueries: {
    episodic: string;
    semantic: string;
    habit: string;
    style: string;
    source: string;
  };
};

export type PersonaTurnPlan = {
  gate: PersonaGateOutput;
  context: PersonaContextGatewayOutput;
};

export type PersonaMemoryTriageIntent =
  | "explicit_memory_request"
  | "explicit_forget_request"
  | "user_preference"
  | "autobiographical_fact"
  | "autobiographical_reasoning"
  | "emotional_salience"
  | "relationship_context"
  | "self_correction_feedback"
  | "durable_product_lesson"
  | "none";

export type PersonaMemoryTriageType =
  | "interaction"
  | "preference"
  | "correction"
  | "lesson"
  | "none";

export type PersonaMemoryTriageSource = "hard_rule" | "llm" | "llm_fallback";

export type PersonaMemoryTriageDecision = {
  confidence: number;
  dedupeKey: string | null;
  memoryIntent: PersonaMemoryTriageIntent;
  memoryType: PersonaMemoryTriageType;
  privacyLevel: PersonaPrivacyLevel;
  reason: string;
  shouldRemember: boolean;
  source: PersonaMemoryTriageSource;
  summary: string | null;
  themes: string[];
};

export type PersonaRuntimeContext = {
  disclosurePolicy: DisclosurePolicy;
  persona: PersonaProfile;
  promptContext: string;
  turnPlan: PersonaTurnPlan;
  workspaceId: string;
};

export type PersonaJsonLlm = (input: {
  systemPrompt: string;
  userPrompt: string;
}) => Promise<unknown>;

// Deferred effect for text embeddings, mirroring PersonaJsonLlm: callers
// inject the provider so retrieval and write paths stay testable and degrade
// to lexical matching when no embedder is configured.
export type PersonaEmbedder = (texts: string[]) => Promise<number[][]>;

export type PersonaAffectVector = {
  arousal: number;
  dominance: number;
  valence: number;
};

export type PersonaMoodEmotionLabelSource =
  | "activated_memory"
  | "context_emotion"
  | "pad_inference"
  | "query_emotion_hint"
  | "query_type";

export type PersonaMoodEmotionLabel = {
  evidence?: string;
  intensity: number;
  label: string;
  memoryId?: string;
  source: PersonaMoodEmotionLabelSource;
};

export type PersonaMoodSignal = {
  affect?: PersonaAffectVector;
  emotionLabels?: PersonaMoodEmotionLabel[];
  id?: string;
  kind:
    | "activated_memory"
    | "context_emotion"
    | "query_emotion_hint"
    | "query_type";
  label?: string;
  queryType?: PersonaQueryType;
  summary?: string;
};

export type PersonaTurnAffectEstimate = {
  emotionLabels: PersonaMoodEmotionLabel[];
  impulse: PersonaAffectVector | null;
  reasons: string[];
  sources: PersonaMoodSignal[];
};

export type PersonaMoodUpdateTrace = {
  baseline: PersonaAffectVector;
  decayFactor: number;
  decayHalfLifeHours: number;
  decayedMood: PersonaAffectVector;
  elapsedMs: number;
  emotionLabels: PersonaMoodEmotionLabel[];
  impulse: PersonaAffectVector | null;
  inertia: number;
  previousMood: PersonaAffectVector | null;
  reasons: string[];
  result: PersonaAffectVector;
  sources: PersonaMoodSignal[];
  version: 1;
};

export type PersonaMemoryCandidate = {
  // How many turns ever activated this memory; repeated recall strengthens
  // the trace (use-dependent consolidation).
  activationCount: number;
  // PAD affect annotation from the emotional salience layer; null when the
  // memory carries no emotion annotation. Drives mood-congruent recall and
  // the per-turn mood update.
  affect: PersonaAffectVector | null;
  // Extra ids this candidate can be addressed by when other memories link to
  // it (e.g. a source chunk also answers to its parent document id). The
  // candidate's own id is always addressable and does not need repeating.
  aliasIds: string[];
  confidence: number;
  createdAt: Date;
  emotions: PersonaEmotionAnnotation[];
  guidance: string[];
  id: string;
  kind: "episode" | "fact" | "belief" | "habit" | "style" | "source";
  // When this memory last fired in a turn; recent recall primes availability.
  lastActivatedAt: Date | null;
  // Outbound references to other memories (supporting episodes, source
  // documents/chunks). Used for one-hop spreading activation.
  linkIds: string[];
  privacyLevel: PersonaPrivacyLevel;
  retrievalBoost: number;
  // Cosine similarity (0..1) against this turn's retrieval query, computed in
  // SQL from stored embeddings. Null when no embedder/embedding is available.
  semanticSimilarity: number | null;
  sourceConfidence: number;
  strength: number;
  summary: string;
  text: string;
  themes: string[];
  voice: string | null;
};

export type PersonaMemoryActivationScoreBreakdown = {
  components: PersonaMemoryRetrievalPayload["candidates"][number]["components"];
  score: number;
};

export type PersonaMemoryRankedEntry = {
  activationScore: number | null;
  availability: number | null;
  baseRank: number | null;
  candidate: PersonaMemoryCandidate;
  components: PersonaMemoryRetrievalPayload["candidates"][number]["components"];
  excludedReason: "privacy" | null;
  query: string;
  rank: number;
  rankBeforeSpreading: number | null;
  spreadingBoost: number;
};

export type PersonaMemorySelectionResult = {
  retrieval: PersonaMemoryRetrievalPayload;
  selected: PersonaMemoryCandidate[];
};

// Identity memory (beliefs, habits, style) is tonic: it stays active every
// turn so the persona keeps a stable point of view. Episodic and source
// memory is phasic: it competes on query relevance within its own budget
// instead of competing with identity memory in one global pool.
export const MEMORY_KIND_BUDGETS: Record<
  PersonaMemoryCandidate["kind"],
  number
> = {
  belief: 4,
  episode: 3,
  fact: 4,
  habit: 2,
  source: 3,
  style: 2,
};
export const MAX_PROMPT_CONTEXT_CHARS = 12_000;
export const MAX_MEMORY_ENTRY_CHARS = 700;
export const MAX_EVIDENCE_SUMMARY_CHARS = 200;
export const MAX_SOURCE_EXCERPT_SUMMARY_CHARS = 620;
export const HINDSIGHT_EXTERNAL_OBSERVATION_PROMPT_CAP = 3;
export const HINDSIGHT_EXTERNAL_SOURCE_PROMPT_CAP = 2;
export const MAX_HINDSIGHT_RETAIN_CONTENT_CHARS = 12_000;
export const MEMORY_TRIAGE_CONFIDENCE_THRESHOLD = 0.62;
export const MEMORY_TRIAGE_SELECTED_MEMORY_CAP = 8;

function normalizePersonaQueryFocusTime(value: unknown): unknown {
  if (!Array.isArray(value)) {
    return value;
  }

  if (value.length === 0) {
    return null;
  }

  const timeEntries = value.map((entry) =>
    typeof entry === "string" ? entry.trim() : null
  );
  if (timeEntries.some((entry) => entry === null)) {
    return value;
  }

  const nonEmptyTimeEntries = timeEntries.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0
  );
  return nonEmptyTimeEntries.length > 0 ? nonEmptyTimeEntries.join(", ") : null;
}

const personaGateSchema = z
  .object({
    answerMode: z.enum(PERSONA_ANSWER_MODES),
    audit: z
      .object({
        confidence: z.number().min(0).max(1),
        matchedSignals: z.array(z.string().min(1)),
        requiresClarification: z.boolean(),
      })
      .strict(),
    neededMemoryTypes: z.array(z.enum(PERSONA_MEMORY_TYPES)),
    queryFocus: z
      .object({
        emotionHints: z.array(z.string()),
        people: z.array(z.string()),
        themes: z.array(z.string()),
        time: z.preprocess(
          normalizePersonaQueryFocusTime,
          z.string().nullable()
        ),
      })
      .strict(),
    queryType: z.enum(PERSONA_QUERY_TYPES),
    safetyFlags: z.array(z.string()),
  })
  .strict();

const personaContextGatewaySchema = z
  .object({
    contextKeys: z
      .object({
        domains: z.array(z.string()),
        emotions: z.array(z.string()),
        entities: z.array(z.string()),
        lifeStages: z.array(z.string()),
        themes: z.array(z.string()),
        timePeriods: z.array(z.string()),
      })
      .strict(),
    retrievalQueries: z
      .object({
        episodic: z.string(),
        habit: z.string(),
        semantic: z.string(),
        source: z.string(),
        style: z.string(),
      })
      .strict(),
  })
  .strict();

export const personaTurnPlanSchema = z
  .object({
    context: personaContextGatewaySchema,
    gate: personaGateSchema,
  })
  .strict();

export const personaMemoryTriageDecisionSchema = z
  .object({
    confidence: z.number().min(0).max(1),
    dedupeKey: z.string().min(1).nullable().default(null),
    memoryIntent: z.enum([
      "explicit_memory_request",
      "explicit_forget_request",
      "user_preference",
      "autobiographical_fact",
      "autobiographical_reasoning",
      "emotional_salience",
      "relationship_context",
      "self_correction_feedback",
      "durable_product_lesson",
      "none",
    ]),
    memoryType: z.enum([
      "interaction",
      "preference",
      "correction",
      "lesson",
      "none",
    ]),
    privacyLevel: z.enum(PERSONA_PRIVACY_LEVELS),
    reason: z.string().min(1),
    shouldRemember: z.boolean(),
    summary: z.string().min(1).nullable(),
    themes: z.array(z.string()).default([]),
  })
  .strict();

export const consolidationOutputSchema = z
  .object({
    beliefs: z
      .array(
        z
          .object({
            // create: a genuinely new belief. reinforce: the interactions
            // re-confirm targetBeliefId. contradict: the interactions
            // contradict targetBeliefId; proposition then carries the revised
            // statement (or repeats the old one when no revision is clear).
            action: z
              .enum(["create", "reinforce", "contradict"])
              .default("create"),
            beliefType: z
              .enum([
                "moral_principle",
                "epistemic_principle",
                "political_belief",
                "aesthetic_preference",
                "relationship_belief",
                "self_concept",
                "professional_norm",
                "conflict_strategy",
              ])
              .default("relationship_belief"),
            confidence: z.number().min(0).max(1),
            domain: z.string().min(1),
            firstPersonForm: z.string().optional(),
            proposition: z.string().min(1),
            strength: z.number().min(0).max(1),
            targetBeliefId: z.string().optional(),
          })
          .strict()
      )
      .default([]),
    episodes: z
      .array(
        z
          .object({
            confidence: z.number().min(0).max(1),
            eventSummary: z.string().min(1),
            title: z.string().min(1),
          })
          .strict()
      )
      .default([]),
    habits: z
      .array(
        z
          .object({
            confidence: z.number().min(0).max(1),
            defaultResponsePattern: z.array(z.string().min(1)).min(1),
            strength: z.number().min(0).max(1),
            trigger: z
              .object({
                description: z.string().min(1),
                type: z.string().min(1),
              })
              .strict(),
          })
          .strict()
      )
      .default([]),
    styleProfiles: z
      .array(
        z
          .object({
            register: z.string().min(1),
            sentenceLength: z.string().min(1),
            toneVector: z.record(z.string(), z.number()).default({}),
          })
          .strict()
      )
      .default([]),
  })
  .strict();
