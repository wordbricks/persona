import { and, asc, desc, eq, or } from "drizzle-orm";

import type { PersonaDatabase as Database } from "../db";
import {
  personaEmotionalSalience,
  personaEpisodeMemories,
  personaFacts,
  personaHabitPatterns,
  personaInteractionMemories,
  personaProfiles,
  personaSemanticBeliefs,
  personaStyleProfiles,
  PERSONA_BELIEF_STANCES,
} from "../schema";
import type {
  NewPersonaEpisodeMemory,
  NewPersonaHabitPattern,
  NewPersonaSemanticBelief,
  NewPersonaStyleProfile,
  PersonaExternalMemoryLocalTargetKind,
  PersonaBeliefStance,
  PersonaBeliefType,
  PersonaEmotionAnnotation,
  PersonaFactObjectType,
  PersonaFactType,
  PersonaMemoryState,
  PersonaEpisodeMemory,
  PersonaHabitPattern,
  PersonaInteractionMemory,
  PersonaPrivacyLevel,
  PersonaProfile,
  PersonaSemanticBelief,
  PersonaStyleProfile,
  PersonaToneVector,
} from "../schema";
import {
  backfillPersonaMemoryEmbeddings,
  beliefMemoryText,
  episodeMemoryText,
  factMemoryText,
  habitMemoryText,
  styleMemoryText,
  tryUpsertPersonaMemoryEmbeddings,
} from "./embeddings";
import type { HindsightPersonaMemoryClient } from "./hindsight";
import { personaLogMessage } from "./logger";
import {
  calculatePersonaMoodUpdate,
  clampScore,
  explainTurnAffect,
  truncateLine,
} from "./mood";
import { loadPersonaProfile } from "./profile";
import {
  PERSONA_CONSOLIDATION_SYSTEM_PROMPT,
  PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT,
  PERSONA_TURN_PLANNER_SYSTEM_PROMPT,
} from "./prompts";
import { resolvePersonaLogger, USER_PREFERENCE_PATTERNS } from "./runtime";
import {
  createPersonaSourceDocument,
  normalizePersonaFactObjectKey,
  scheduleHindsightRetain,
} from "./source-ingestion";
import {
  consolidationOutputSchema,
  MAX_HINDSIGHT_RETAIN_CONTENT_CHARS,
  MAX_MEMORY_ENTRY_CHARS,
  PERSONA_ANSWER_MODES,
  PERSONA_MEMORY_TYPES,
  PERSONA_QUERY_TYPES,
} from "./types";
import type {
  DisclosurePolicy,
  PersonaAffectVector,
  PersonaContextGatewayOutput,
  PersonaEmbedder,
  PersonaJsonLlm,
  PersonaMemoryCandidate,
  PersonaMemorySelectionResult,
  PersonaMemoryTriageDecision,
  PersonaMemoryTriageIntent,
  PersonaMemoryTriageSource,
  PersonaMemoryTriageType,
  PersonaMoodEmotionLabel,
  PersonaMoodSignal,
  PersonaMoodUpdateTrace,
  PersonaLogger,
  PersonaQueryType,
  PersonaRuntimeContext,
  PersonaTurnAffectEstimate,
  PersonaTurnPlan,
} from "./types";

export {
  createHindsightPersonaMemoryClient,
  createHindsightPersonaMemoryConfig,
  createNoopHindsightPersonaMemoryClient,
  hindsightPersonaBankId,
  hindsightPersonaTags,
} from "./hindsight";
export type {
  HindsightPersonaMemoryClient,
  HindsightPersonaMemoryConfig,
  HindsightPersonaMemoryEnv,
  HindsightRecallInput,
  HindsightRecallResult,
  HindsightReflectInput,
  HindsightReflectResult,
  HindsightRetainInput,
  HindsightRetainResult,
} from "./hindsight";
export {
  backfillPersonaMemoryEmbeddings,
  createOpenAiPersonaEmbedder,
  hashPersonaMemoryText,
  normalizePersonaEmbeddingText,
  upsertPersonaMemoryEmbeddings,
} from "./embeddings";
export type { PersonaMemoryEmbeddingEntry } from "./embeddings";
export * from "./delete-profile";
export {
  calculatePersonaMoodUpdate,
  estimateTurnAffect,
  explainTurnAffect,
  PERSONA_MOOD_BASELINE,
  updatePersonaMood,
} from "./mood";
export {
  copyPersonaProfile,
  forgetPersonaLayerMemory,
  listPersonaAliases,
  listPublicPersonaProfiles,
  loadPersonaProfile,
  publishPersonaProfile,
  upsertPersonaAlias,
  upsertPersonaProfile,
} from "./profile";
export {
  PERSONA_CONSOLIDATION_SYSTEM_PROMPT,
  PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT,
  PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT,
  PERSONA_TURN_PLANNER_SYSTEM_PROMPT,
} from "./prompts";
export {
  calculateMemoryAvailability,
  normalizeCosineSimilarity,
  selectPersonaMemories,
  selectPersonaMemoriesWithScores,
} from "./selection";
export {
  evaluatePostResponseInteractionMemory,
  formatWorkspacePrompt,
  planPersonaTurnWithLlm,
  preparePersonaRuntimeContext,
  recallPersonaMemoriesForCue,
  recordPostResponsePersonaMemoryReview,
  shouldRecordInteractionMemory,
  triageInteractionMemoryWithLlm,
  triagePostResponseInteractionMemoryWithLlm,
} from "./runtime";
export {
  activatePersonaDraftMemory,
  chunkPersonaSourceText,
  draftPersonaMemoriesFromSourceDocument,
  formatPersonaSourceExcerptForQuery,
  hashPersonaSourceContent,
  ingestPersonaSourceDocument,
} from "./source-ingestion";
export type {
  PersonaDraftMemorySummary,
  PersonaSourceDraftBelief,
  PersonaSourceDraftEpisode,
  PersonaSourceDraftFact,
  PersonaSourceDraftHabit,
  PersonaSourceDraftMemoryInput,
  PersonaSourceDraftMemoryResult,
  PersonaSourceDraftStyleProfile,
  PersonaSourceIngestionResult,
  PersonaSourceTextChunk,
} from "./source-ingestion";
export { PERSONA_ANSWER_MODES, PERSONA_MEMORY_TYPES, PERSONA_QUERY_TYPES };
export type {
  DisclosurePolicy,
  PersonaAffectVector,
  PersonaAnswerMode,
  PersonaContextGatewayOutput,
  PersonaEmbedder,
  PersonaGateOutput,
  PersonaJsonLlm,
  PersonaMemoryCandidate,
  PersonaMemorySelectionResult,
  PersonaMemoryTriageDecision,
  PersonaMemoryTriageIntent,
  PersonaMemoryTriageSource,
  PersonaMemoryTriageType,
  PersonaLogger,
  PersonaMemoryType,
  PersonaMoodEmotionLabel,
  PersonaMoodEmotionLabelSource,
  PersonaMoodSignal,
  PersonaMoodUpdateTrace,
  PersonaQueryType,
  PersonaRuntimeContext,
  PersonaTurnAffectEstimate,
  PersonaTurnPlan,
} from "./types";

// Bounded Hebbian-style belief updates: repeated confirmation asymptotically
// approaches certainty, contradiction decays conviction, and crossing the
// conflict threshold flips the lifecycle state instead of deleting history
// (reconsolidation, not overwrite).
const BELIEF_REINFORCEMENT_RATE = 0.3;
const BELIEF_CONTRADICTION_RATE = 0.45;
const BELIEF_CONFLICT_THRESHOLD = 0.35;

export function applyBeliefReinforcement(
  belief: Pick<PersonaSemanticBelief, "confidence" | "strength">,
  input: { evidenceConfidence: number }
): { confidence: number; strength: number } {
  const rate = BELIEF_REINFORCEMENT_RATE * clampScore(input.evidenceConfidence);
  return {
    confidence: clampScore(belief.confidence + rate * (1 - belief.confidence)),
    strength: clampScore(belief.strength + rate * (1 - belief.strength)),
  };
}

export function applyBeliefContradiction(
  belief: Pick<PersonaSemanticBelief, "confidence" | "strength">,
  input: { evidenceConfidence: number }
): { confidence: number; state: PersonaMemoryState; strength: number } {
  const rate = BELIEF_CONTRADICTION_RATE * clampScore(input.evidenceConfidence);
  const strength = clampScore(belief.strength * (1 - rate));
  return {
    confidence: clampScore(belief.confidence * (1 - rate)),
    state: strength < BELIEF_CONFLICT_THRESHOLD ? "conflicted" : "active",
    strength,
  };
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is string => typeof entry === "string");
}

function buildConsolidationUserPrompt(input: {
  existingBeliefs: Pick<
    PersonaSemanticBelief,
    "domain" | "id" | "proposition" | "strength"
  >[];
  interactions: {
    id: string;
    summary: string;
    newUserPreference: string | null;
  }[];
  persona: PersonaProfile;
  reflection?: string | null;
}) {
  return [
    `personaKey: ${input.persona.personaKey}`,
    `displayName: ${input.persona.displayName}`,
    "",
    "Existing beliefs:",
    ...(input.existingBeliefs.length > 0
      ? input.existingBeliefs.map((belief) =>
          truncateLine(
            `- ${belief.id} [${belief.domain}] strength=${belief.strength.toFixed(2)}: ${belief.proposition}`,
            MAX_MEMORY_ENTRY_CHARS
          )
        )
      : ["- none"]),
    "",
    "Interaction memories:",
    ...input.interactions.map(
      (entry) =>
        `- ${entry.id}: ${entry.summary}${
          entry.newUserPreference
            ? ` preference=${entry.newUserPreference}`
            : ""
        }`
    ),
    ...(input.reflection
      ? [
          "",
          "External reflection evidence:",
          truncateLine(input.reflection, MAX_MEMORY_ENTRY_CHARS),
          "Use this only as supporting evidence. Local consolidation policy and the listed interaction ids remain authoritative.",
        ]
      : []),
  ].join("\n");
}

function hindsightScopeForLayerMemory(input: {
  memoryKind: "episode" | "fact" | "belief" | "habit" | "style" | "interaction";
  privacyLevel: PersonaPrivacyLevel;
  summary: string;
  title: string;
  userId?: string | null;
}): "persona_global" | "persona_user" | null {
  if (input.memoryKind === "interaction") {
    return input.userId ? "persona_user" : null;
  }
  if (input.privacyLevel === "sensitive") {
    return null;
  }
  if (input.privacyLevel === "private" && input.userId) {
    return "persona_user";
  }
  if (input.memoryKind === "episode" && input.userId) {
    return "persona_user";
  }
  if (
    input.memoryKind === "fact" &&
    input.userId &&
    USER_PREFERENCE_PATTERNS.some((pattern) =>
      pattern.test(`${input.title}\n${input.summary}`)
    )
  ) {
    return "persona_user";
  }
  return "persona_global";
}

function scheduleHindsightLayerMemoryRetain(input: {
  content: Record<string, unknown>;
  db: Database;
  defer?: (promise: Promise<unknown>) => void;
  hindsight?: HindsightPersonaMemoryClient | null;
  logger?: PersonaLogger;
  localTargetId: string;
  memoryKind: "episode" | "fact" | "belief" | "habit" | "style" | "interaction";
  organizationId: string;
  persona: PersonaProfile;
  privacyLevel: PersonaPrivacyLevel;
  summary: string;
  title: string;
  userId?: string | null;
}): void {
  const scope = hindsightScopeForLayerMemory({
    memoryKind: input.memoryKind,
    privacyLevel: input.privacyLevel,
    summary: input.summary,
    title: input.title,
    userId: input.userId,
  });
  if (!scope) {
    return;
  }
  const themes = [
    ...readStringArray(input.content.themes),
    ...readStringArray(input.content.domains),
  ];
  const content = `${input.title}\n${input.summary}`.slice(
    0,
    MAX_HINDSIGHT_RETAIN_CONTENT_CHARS
  );
  scheduleHindsightRetain({
    db: input.db,
    defer: input.defer,
    hindsight: input.hindsight,
    logger: input.logger,
    localTargetId: input.localTargetId,
    localTargetKind: input.memoryKind,
    metadata: {
      memoryKind: input.memoryKind,
      source: "rememberPersonaLayerMemory",
    },
    organizationId: input.organizationId,
    personaId: input.persona.id,
    privacyLevel: input.privacyLevel,
    retainInput: {
      content,
      context: `Durable persona ${input.memoryKind} memory`,
      documentId: `persona_${input.memoryKind}_${input.localTargetId}`,
      organizationId: input.organizationId,
      personaId: input.persona.id,
      personaKey: input.persona.personaKey,
      privacyLevel: input.privacyLevel,
      scope,
      tags: [`memory_kind_${input.memoryKind}`],
      themes,
      timestamp: new Date(),
      userId: scope === "persona_user" ? input.userId : null,
    },
    userId: scope === "persona_user" ? input.userId : null,
  });
}

async function maybeReflectHindsightConsolidation(input: {
  hindsight?: HindsightPersonaMemoryClient | null;
  interactions: PersonaInteractionMemory[];
  logger?: PersonaLogger;
  organizationId: string;
  persona: PersonaProfile;
  userId?: string | null;
}): Promise<string | null> {
  if (!input.hindsight || input.interactions.length === 0) {
    return null;
  }
  const logger = resolvePersonaLogger(input.logger);
  try {
    const response = await input.hindsight.reflect({
      context: input.interactions
        .map(
          (interaction) =>
            `${interaction.id}: ${interaction.interactionSummary}`
        )
        .join("\n")
        .slice(0, MAX_HINDSIGHT_RETAIN_CONTENT_CHARS),
      organizationId: input.organizationId,
      personaId: input.persona.id,
      personaKey: input.persona.personaKey,
      query:
        "What stable persona memory, contradictions, or preference changes are supported by these interaction memories?",
      scope: input.userId ? "persona_user" : "persona_global",
      tags: ["consolidation"],
      userId: input.userId ?? null,
    });
    return response.text;
  } catch (error) {
    logger.warn(
      personaLogMessage("[persona-memory] hindsight reflection failed:", error)
    );
    return null;
  }
}

function scheduleHindsightConsolidatedMemoryRetain(input: {
  content: string;
  db: Database;
  defer?: (promise: Promise<unknown>) => void;
  hindsight?: HindsightPersonaMemoryClient | null;
  logger?: PersonaLogger;
  localTargetId: string;
  localTargetKind: Exclude<
    PersonaExternalMemoryLocalTargetKind,
    "interaction" | "source_document" | "source_chunk"
  >;
  organizationId: string;
  persona: PersonaProfile;
  title: string;
  userId?: string | null;
}): void {
  const scope = input.userId ? "persona_user" : "persona_global";
  const privacyLevel = input.userId ? "private" : "internal";
  scheduleHindsightRetain({
    db: input.db,
    defer: input.defer,
    hindsight: input.hindsight,
    logger: input.logger,
    localTargetId: input.localTargetId,
    localTargetKind: input.localTargetKind,
    metadata: {
      memoryKind: input.localTargetKind,
      source: "consolidatePersonaMemoryScope",
    },
    organizationId: input.organizationId,
    personaId: input.persona.id,
    privacyLevel,
    retainInput: {
      content: `${input.title}\n${input.content}`.slice(
        0,
        MAX_HINDSIGHT_RETAIN_CONTENT_CHARS
      ),
      context: `Consolidated persona ${input.localTargetKind} memory`,
      documentId: `persona_${input.localTargetKind}_${input.localTargetId}`,
      organizationId: input.organizationId,
      personaId: input.persona.id,
      personaKey: input.persona.personaKey,
      privacyLevel,
      scope,
      tags: ["consolidated_memory", `memory_kind_${input.localTargetKind}`],
      themes: [],
      timestamp: new Date(),
      userId: input.userId ?? null,
    },
    userId: input.userId ?? null,
  });
}

function readFiniteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const PERSONA_TONE_KEYS = [
  "warmth",
  "formality",
  "humor",
  "directness",
  "humility",
  "firmness",
  "defensiveness",
] as const;

export function parsePersonaToneVector(value: unknown): PersonaToneVector {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  const record = value as Record<string, unknown>;
  const tone: PersonaToneVector = {};
  for (const key of PERSONA_TONE_KEYS) {
    const raw = readFiniteNumber(record[key]);
    if (raw !== null) {
      tone[key] = clampScore(raw);
    }
  }
  return tone;
}

export function parsePersonaBeliefStance(value: unknown): PersonaBeliefStance {
  return typeof value === "string" &&
    (PERSONA_BELIEF_STANCES as readonly string[]).includes(value)
    ? (value as PersonaBeliefStance)
    : "support";
}

export function parsePersonaLexicalPreferences(value: unknown): {
  uses?: string[];
  avoids?: string[];
} {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  const record = value as Record<string, unknown>;
  const uses = readStringArray(record.uses);
  const avoids = readStringArray(record.avoids);
  return {
    ...(uses.length > 0 ? { uses } : {}),
    ...(avoids.length > 0 ? { avoids } : {}),
  };
}

export type PersonaEpisodeAffectInput = {
  arousal: number;
  dominance: number;
  emotions: PersonaEmotionAnnotation[];
  retrievalBoost: number;
  salienceScore: number;
  selfRelevance: number;
  valence: number;
};

// Parses an episode's emotional annotation (PAD + salience) from memory
// content. Valence and arousal are required; everything else falls back to
// neutral defaults. The resulting persona_emotional_salience row is what
// powers mood updates, mood-congruent recall, and retrieval boosts.
export function parsePersonaEpisodeAffect(
  value: unknown
): PersonaEpisodeAffectInput | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const rawValence = readFiniteNumber(record.valence);
  const rawArousal = readFiniteNumber(record.arousal);
  if (rawValence === null || rawArousal === null) {
    return null;
  }
  const valence = Math.max(-1, Math.min(1, rawValence));
  const arousal = clampScore(rawArousal);
  const emotions: PersonaEmotionAnnotation[] = Array.isArray(record.emotions)
    ? record.emotions.flatMap((entry) => {
        if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
          return [];
        }
        const annotation = entry as Record<string, unknown>;
        const emotion =
          typeof annotation.emotion === "string"
            ? annotation.emotion.trim()
            : "";
        const intensity = readFiniteNumber(annotation.intensity);
        if (emotion.length === 0 || intensity === null) {
          return [];
        }
        return [
          {
            emotion,
            intensity: clampScore(intensity),
            ...(typeof annotation.evidence === "string"
              ? { evidence: annotation.evidence }
              : {}),
          },
        ];
      })
    : [];
  const rawBoost = readFiniteNumber(record.retrievalBoost);
  return {
    arousal,
    dominance: clampScore(readFiniteNumber(record.dominance) ?? 0.5),
    emotions,
    retrievalBoost:
      rawBoost === null ? 1 : Math.max(0.5, Math.min(2, rawBoost)),
    salienceScore: clampScore(
      readFiniteNumber(record.salienceScore) ??
        Math.max(arousal, Math.abs(valence))
    ),
    selfRelevance: clampScore(readFiniteNumber(record.selfRelevance) ?? 0.5),
    valence,
  };
}

export async function rememberPersonaLayerMemory(
  db: Database,
  input: {
    confidence?: number | null;
    content?: Record<string, unknown> | null;
    defer?: (promise: Promise<unknown>) => void;
    embed?: PersonaEmbedder;
    hindsight?: HindsightPersonaMemoryClient | null;
    logger?: PersonaLogger;
    memoryKind:
      | "episode"
      | "fact"
      | "belief"
      | "habit"
      | "style"
      | "interaction";
    organizationId: string;
    personaKey: string;
    privacyLevel?: PersonaPrivacyLevel | null;
    summary: string;
    title: string;
    updatedByUserId?: string | null;
    userId?: string | null;
  }
): Promise<{
  id: string;
  memoryKind: "episode" | "fact" | "belief" | "habit" | "style" | "interaction";
}> {
  const persona = await loadPersonaProfile(db, {
    organizationId: input.organizationId,
    personaKey: input.personaKey,
  });
  const confidence = clampScore(input.confidence ?? 0.82);
  const privacyLevel = input.privacyLevel ?? "internal";
  const content = input.content ?? {};
  const now = new Date();
  const sourceDocument = await createPersonaSourceDocument(db, {
    authorUserId: input.updatedByUserId,
    embed: input.embed,
    logger: input.logger,
    organizationId: input.organizationId,
    persona,
    privacyLevel,
    rawText: `${input.title}\n${input.summary}`,
    sourceType: "seed",
    title: input.title,
  });

  if (input.memoryKind === "episode") {
    const [episode] = await db
      .insert(personaEpisodeMemories)
      .values({
        confidence,
        eventSummary: input.summary,
        firstPersonRecollection:
          typeof content.firstPersonRecollection === "string"
            ? content.firstPersonRecollection
            : null,
        organizationId: input.organizationId,
        personaId: persona.id,
        privacyLevel,
        sourceRefs: [{ sourceDocumentId: sourceDocument.id }],
        themes: readStringArray(content.themes),
        time:
          content.time && typeof content.time === "object"
            ? (content.time as NewPersonaEpisodeMemory["time"])
            : {},
        title: input.title,
        updatedAt: now,
      })
      .returning();
    if (episode) {
      // Optional PAD annotation: feeds mood updates, mood-congruent recall,
      // and salience-based retrieval boosts for this episode.
      const affect = parsePersonaEpisodeAffect(content.affect);
      if (affect) {
        await db.insert(personaEmotionalSalience).values({
          arousal: affect.arousal,
          confidence,
          dominance: affect.dominance,
          emotions: affect.emotions,
          episodeMemoryId: episode.id,
          organizationId: input.organizationId,
          personaId: persona.id,
          retrievalBoost: affect.retrievalBoost,
          salienceScore: affect.salienceScore,
          selfRelevance: affect.selfRelevance,
          updatedAt: now,
          valence: affect.valence,
        });
      }
      await tryUpsertPersonaMemoryEmbeddings(db, {
        embed: input.embed,
        entries: [
          {
            organizationId: input.organizationId,
            personaId: persona.id,
            targetId: episode.id,
            targetKind: "episode",
            text: episodeMemoryText(episode),
          },
        ],
        logger: input.logger,
      });
      scheduleHindsightLayerMemoryRetain({
        content,
        db,
        defer: input.defer,
        hindsight: input.hindsight,
        logger: input.logger,
        localTargetId: episode.id,
        memoryKind: "episode",
        organizationId: input.organizationId,
        persona,
        privacyLevel,
        summary: input.summary,
        title: input.title,
        userId: input.userId,
      });
    }
    return { id: episode?.id ?? "", memoryKind: "episode" };
  }

  if (input.memoryKind === "belief") {
    const [belief] = await db
      .insert(personaSemanticBeliefs)
      .values({
        beliefType:
          typeof content.beliefType === "string"
            ? (content.beliefType as PersonaBeliefType)
            : "relationship_belief",
        confidence,
        domain: typeof content.domain === "string" ? content.domain : "general",
        exceptions: readStringArray(content.exceptions),
        firstPersonForm:
          typeof content.firstPersonForm === "string"
            ? content.firstPersonForm
            : null,
        organizationId: input.organizationId,
        personaId: persona.id,
        privacyLevel,
        proposition: input.summary,
        stance: parsePersonaBeliefStance(content.stance),
        strength: confidence,
        supportingSourceIds: [sourceDocument.id],
        updatedAt: now,
      })
      .returning();
    if (belief) {
      await tryUpsertPersonaMemoryEmbeddings(db, {
        embed: input.embed,
        entries: [
          {
            organizationId: input.organizationId,
            personaId: persona.id,
            targetId: belief.id,
            targetKind: "belief",
            text: beliefMemoryText(belief),
          },
        ],
        logger: input.logger,
      });
      scheduleHindsightLayerMemoryRetain({
        content,
        db,
        defer: input.defer,
        hindsight: input.hindsight,
        logger: input.logger,
        localTargetId: belief.id,
        memoryKind: "belief",
        organizationId: input.organizationId,
        persona,
        privacyLevel,
        summary: input.summary,
        title: input.title,
        userId: input.userId,
      });
    }
    return { id: belief?.id ?? "", memoryKind: "belief" };
  }

  if (input.memoryKind === "fact") {
    const claimText =
      typeof content.claimText === "string" && content.claimText.trim()
        ? content.claimText.trim()
        : input.summary;
    const objectName =
      typeof content.objectName === "string" && content.objectName.trim()
        ? content.objectName.trim()
        : input.title;
    const [fact] = await db
      .insert(personaFacts)
      .values({
        claimText,
        confidence,
        evidenceSpan:
          typeof content.evidenceSpan === "string"
            ? content.evidenceSpan
            : null,
        factType:
          typeof content.factType === "string"
            ? (content.factType as PersonaFactType)
            : "other",
        firstPersonForm:
          typeof content.firstPersonForm === "string"
            ? content.firstPersonForm
            : null,
        objectKey:
          typeof content.objectKey === "string" && content.objectKey.trim()
            ? content.objectKey.trim()
            : normalizePersonaFactObjectKey(objectName),
        objectName,
        objectType:
          typeof content.objectType === "string"
            ? (content.objectType as PersonaFactObjectType)
            : "text",
        organizationId: input.organizationId,
        personaId: persona.id,
        privacyLevel,
        sourceDocumentId: sourceDocument.id,
        sourceRefs: [{ sourceDocumentId: sourceDocument.id }],
        updatedAt: now,
      })
      .returning();
    if (fact) {
      await tryUpsertPersonaMemoryEmbeddings(db, {
        embed: input.embed,
        entries: [
          {
            organizationId: input.organizationId,
            personaId: persona.id,
            targetId: fact.id,
            targetKind: "fact",
            text: factMemoryText(fact),
          },
        ],
        logger: input.logger,
      });
      scheduleHindsightLayerMemoryRetain({
        content,
        db,
        defer: input.defer,
        hindsight: input.hindsight,
        logger: input.logger,
        localTargetId: fact.id,
        memoryKind: "fact",
        organizationId: input.organizationId,
        persona,
        privacyLevel,
        summary: input.summary,
        title: input.title,
        userId: input.userId,
      });
    }
    return { id: fact?.id ?? "", memoryKind: "fact" };
  }

  if (input.memoryKind === "habit") {
    const [habit] = await db
      .insert(personaHabitPatterns)
      .values({
        avoidPatterns: readStringArray(content.avoidPatterns),
        confidence,
        defaultResponsePattern:
          readStringArray(content.defaultResponsePattern).length > 0
            ? readStringArray(content.defaultResponsePattern)
            : [input.summary],
        organizationId: input.organizationId,
        personaId: persona.id,
        rhetoricalMoves: readStringArray(content.rhetoricalMoves),
        strength: confidence,
        tone: parsePersonaToneVector(content.tone),
        trigger: {
          description:
            typeof content.triggerDescription === "string"
              ? content.triggerDescription
              : input.title,
          type:
            typeof content.triggerType === "string"
              ? content.triggerType
              : "general",
        },
        updatedAt: now,
      })
      .returning();
    if (habit) {
      await tryUpsertPersonaMemoryEmbeddings(db, {
        embed: input.embed,
        entries: [
          {
            organizationId: input.organizationId,
            personaId: persona.id,
            targetId: habit.id,
            targetKind: "habit",
            text: habitMemoryText(habit),
          },
        ],
        logger: input.logger,
      });
      scheduleHindsightLayerMemoryRetain({
        content,
        db,
        defer: input.defer,
        hindsight: input.hindsight,
        logger: input.logger,
        localTargetId: habit.id,
        memoryKind: "habit",
        organizationId: input.organizationId,
        persona,
        privacyLevel,
        summary: input.summary,
        title: input.title,
        userId: input.userId,
      });
    }
    return { id: habit?.id ?? "", memoryKind: "habit" };
  }

  if (input.memoryKind === "style") {
    const [style] = await db
      .insert(personaStyleProfiles)
      .values({
        avoidPhrases: readStringArray(content.avoidPhrases),
        commonPhrases: readStringArray(content.commonPhrases),
        lexicalPreferences: parsePersonaLexicalPreferences(
          content.lexicalPreferences
        ),
        organizationId: input.organizationId,
        personaId: persona.id,
        preferredRhetoricalMoves: readStringArray(
          content.preferredRhetoricalMoves
        ),
        register:
          typeof content.register === "string" ? content.register : "neutral",
        sentenceLength:
          typeof content.sentenceLength === "string"
            ? content.sentenceLength
            : "medium",
        toneVector: parsePersonaToneVector(content.toneVector),
        updatedAt: now,
      })
      .returning();
    if (style) {
      await tryUpsertPersonaMemoryEmbeddings(db, {
        embed: input.embed,
        entries: [
          {
            organizationId: input.organizationId,
            personaId: persona.id,
            targetId: style.id,
            targetKind: "style",
            text: styleMemoryText(style),
          },
        ],
        logger: input.logger,
      });
      scheduleHindsightLayerMemoryRetain({
        content,
        db,
        defer: input.defer,
        hindsight: input.hindsight,
        logger: input.logger,
        localTargetId: style.id,
        memoryKind: "style",
        organizationId: input.organizationId,
        persona,
        privacyLevel,
        summary: input.summary,
        title: input.title,
        userId: input.userId,
      });
    }
    return { id: style?.id ?? "", memoryKind: "style" };
  }

  const [interaction] = await db
    .insert(personaInteractionMemories)
    .values({
      interactionSummary: input.summary,
      newPersonaMemory: content,
      organizationId: input.organizationId,
      personaId: persona.id,
      shouldConsolidate: true,
      state: "consolidation_candidate",
      userId: input.userId ?? null,
    })
    .returning();
  if (interaction) {
    scheduleHindsightLayerMemoryRetain({
      content,
      db,
      defer: input.defer,
      hindsight: input.hindsight,
      logger: input.logger,
      localTargetId: interaction.id,
      memoryKind: "interaction",
      organizationId: input.organizationId,
      persona,
      privacyLevel,
      summary: input.summary,
      title: input.title,
      userId: input.userId,
    });
  }
  return { id: interaction?.id ?? "", memoryKind: "interaction" };
}

export async function consolidatePersonaMemoryScope(input: {
  consolidate: PersonaJsonLlm;
  db: Database;
  defer?: (promise: Promise<unknown>) => void;
  embed?: PersonaEmbedder;
  hindsight?: HindsightPersonaMemoryClient | null;
  logger?: PersonaLogger;
  organizationId: string;
  personaKey: string;
  userId?: string | null;
}): Promise<{
  contradictedBeliefs: number;
  createdBeliefs: number;
  createdEpisodes: number;
  createdHabits: number;
  createdStyleProfiles: number;
  personaKey: string;
  reinforcedBeliefs: number;
}> {
  const persona = await loadPersonaProfile(input.db, {
    organizationId: input.organizationId,
    personaKey: input.personaKey,
  });
  const interactions = await input.db
    .select()
    .from(personaInteractionMemories)
    .where(
      and(
        eq(personaInteractionMemories.organizationId, input.organizationId),
        eq(personaInteractionMemories.personaId, persona.id),
        input.userId
          ? eq(personaInteractionMemories.userId, input.userId)
          : undefined,
        eq(personaInteractionMemories.state, "consolidation_candidate"),
        eq(personaInteractionMemories.shouldConsolidate, true)
      )
    )
    .orderBy(asc(personaInteractionMemories.createdAt))
    .limit(50);

  if (interactions.length === 0) {
    return {
      contradictedBeliefs: 0,
      createdBeliefs: 0,
      createdEpisodes: 0,
      createdHabits: 0,
      createdStyleProfiles: 0,
      personaKey: persona.personaKey,
      reinforcedBeliefs: 0,
    };
  }

  // Strongest beliefs go to the consolidator so repeated evidence reinforces
  // or revises them instead of accumulating near-duplicates.
  const existingBeliefs = await input.db
    .select()
    .from(personaSemanticBeliefs)
    .where(
      and(
        eq(personaSemanticBeliefs.organizationId, input.organizationId),
        eq(personaSemanticBeliefs.personaId, persona.id),
        eq(personaSemanticBeliefs.state, "active")
      )
    )
    .orderBy(desc(personaSemanticBeliefs.strength))
    .limit(100);
  const existingBeliefById = new Map(
    existingBeliefs.map((belief) => [belief.id, belief])
  );
  const reflection = await maybeReflectHindsightConsolidation({
    hindsight: input.hindsight,
    interactions,
    logger: input.logger,
    organizationId: input.organizationId,
    persona,
    userId: input.userId,
  });

  const output = await input.consolidate({
    systemPrompt: PERSONA_CONSOLIDATION_SYSTEM_PROMPT,
    userPrompt: buildConsolidationUserPrompt({
      existingBeliefs,
      interactions: interactions.map((interaction) => ({
        id: interaction.id,
        newUserPreference: interaction.newUserPreference,
        summary: interaction.interactionSummary,
      })),
      persona,
      reflection,
    }),
  });
  const parsed = consolidationOutputSchema.safeParse(output);
  if (!parsed.success) {
    throw new Error(
      `Persona consolidation returned invalid JSON: ${JSON.stringify(parsed.error.issues)}`
    );
  }

  const sourceIds = interactions.map((interaction) => interaction.id);
  const nowIso = new Date().toISOString();

  type BeliefEntry = (typeof parsed.data.beliefs)[number];
  type BeliefPlan =
    | { entry: BeliefEntry; kind: "create" }
    | {
        entry: BeliefEntry;
        kind: "contradict" | "reinforce";
        target: PersonaSemanticBelief;
      };
  const beliefPlans: BeliefPlan[] = parsed.data.beliefs.map((entry) => {
    const target = entry.targetBeliefId
      ? existingBeliefById.get(entry.targetBeliefId)
      : undefined;
    if (entry.action !== "create" && !target) {
      const logger = resolvePersonaLogger(input.logger);
      logger.warn(
        `[persona-memory] consolidation referenced unknown belief ${entry.targetBeliefId ?? "(missing)"}; treating as create.`
      );
    }
    if (entry.action === "reinforce" && target) {
      return { entry, kind: "reinforce", target };
    }
    if (entry.action === "contradict" && target) {
      return { entry, kind: "contradict", target };
    }
    return { entry, kind: "create" };
  });

  const newBeliefFromEntry = (
    entry: BeliefEntry
  ): NewPersonaSemanticBelief => ({
    beliefType: entry.beliefType,
    confidence: entry.confidence,
    domain: entry.domain,
    firstPersonForm: entry.firstPersonForm ?? null,
    organizationId: input.organizationId,
    personaId: persona.id,
    privacyLevel: "internal",
    proposition: entry.proposition,
    stance: "support",
    strength: entry.strength,
    supportingSourceIds: sourceIds,
  });

  // A contradiction with a genuinely revised statement supersedes the old
  // belief and creates the revision; same-statement contradictions only
  // weaken the old belief in place.
  const isRevision = (
    plan: BeliefPlan
  ): plan is BeliefPlan & {
    kind: "contradict";
    target: PersonaSemanticBelief;
  } =>
    plan.kind === "contradict" &&
    plan.entry.proposition.trim() !== plan.target.proposition.trim();

  const beliefValues: NewPersonaSemanticBelief[] = [
    ...beliefPlans
      .filter((plan) => plan.kind === "create")
      .map((plan) => newBeliefFromEntry(plan.entry)),
    ...beliefPlans.filter(isRevision).map((plan) => ({
      ...newBeliefFromEntry(plan.entry),
      stance: "revised" as const,
    })),
  ];
  const episodeValues: NewPersonaEpisodeMemory[] = parsed.data.episodes.map(
    (episode) => ({
      confidence: episode.confidence,
      eventSummary: episode.eventSummary,
      organizationId: input.organizationId,
      personaId: persona.id,
      privacyLevel: "internal",
      sourceRefs: sourceIds.map((id) => ({ quoteSpan: id })),
      title: episode.title,
    })
  );
  const habitValues: NewPersonaHabitPattern[] = parsed.data.habits.map(
    (habit) => ({
      confidence: habit.confidence,
      defaultResponsePattern: habit.defaultResponsePattern,
      organizationId: input.organizationId,
      personaId: persona.id,
      strength: habit.strength,
      supportingExampleIds: sourceIds,
      trigger: habit.trigger,
    })
  );
  const styleValues: NewPersonaStyleProfile[] = parsed.data.styleProfiles.map(
    (style) => ({
      organizationId: input.organizationId,
      personaId: persona.id,
      register: style.register,
      sentenceLength: style.sentenceLength,
      toneVector: style.toneVector,
    })
  );

  const [createdBeliefs, createdEpisodes, createdHabits, createdStyleProfiles] =
    await Promise.all([
      beliefValues.length > 0
        ? input.db
            .insert(personaSemanticBeliefs)
            .values(beliefValues)
            .returning()
        : Promise.resolve([] as PersonaSemanticBelief[]),
      episodeValues.length > 0
        ? input.db
            .insert(personaEpisodeMemories)
            .values(episodeValues)
            .returning()
        : Promise.resolve([] as PersonaEpisodeMemory[]),
      habitValues.length > 0
        ? input.db.insert(personaHabitPatterns).values(habitValues).returning()
        : Promise.resolve([] as PersonaHabitPattern[]),
      styleValues.length > 0
        ? input.db.insert(personaStyleProfiles).values(styleValues).returning()
        : Promise.resolve([] as PersonaStyleProfile[]),
    ]);

  await tryUpsertPersonaMemoryEmbeddings(input.db, {
    embed: input.embed,
    entries: [
      ...createdEpisodes.map((episode) => ({
        organizationId: input.organizationId,
        personaId: persona.id,
        targetId: episode.id,
        targetKind: "episode" as const,
        text: episodeMemoryText(episode),
      })),
      ...createdBeliefs.map((belief) => ({
        organizationId: input.organizationId,
        personaId: persona.id,
        targetId: belief.id,
        targetKind: "belief" as const,
        text: beliefMemoryText(belief),
      })),
      ...createdHabits.map((habit) => ({
        organizationId: input.organizationId,
        personaId: persona.id,
        targetId: habit.id,
        targetKind: "habit" as const,
        text: habitMemoryText(habit),
      })),
      ...createdStyleProfiles.map((style) => ({
        organizationId: input.organizationId,
        personaId: persona.id,
        targetId: style.id,
        targetKind: "style" as const,
        text: styleMemoryText(style),
      })),
    ],
    logger: input.logger,
  });
  for (const episode of createdEpisodes) {
    scheduleHindsightConsolidatedMemoryRetain({
      content: episode.eventSummary,
      db: input.db,
      defer: input.defer,
      hindsight: input.hindsight,
      logger: input.logger,
      localTargetId: episode.id,
      localTargetKind: "episode",
      organizationId: input.organizationId,
      persona,
      title: episode.title,
      userId: input.userId,
    });
  }
  for (const belief of createdBeliefs) {
    scheduleHindsightConsolidatedMemoryRetain({
      content: beliefMemoryText(belief),
      db: input.db,
      defer: input.defer,
      hindsight: input.hindsight,
      logger: input.logger,
      localTargetId: belief.id,
      localTargetKind: "belief",
      organizationId: input.organizationId,
      persona,
      title: belief.domain,
      userId: input.userId,
    });
  }
  for (const habit of createdHabits) {
    scheduleHindsightConsolidatedMemoryRetain({
      content: habitMemoryText(habit),
      db: input.db,
      defer: input.defer,
      hindsight: input.hindsight,
      logger: input.logger,
      localTargetId: habit.id,
      localTargetKind: "habit",
      organizationId: input.organizationId,
      persona,
      title: habit.trigger.description,
      userId: input.userId,
    });
  }
  for (const style of createdStyleProfiles) {
    scheduleHindsightConsolidatedMemoryRetain({
      content: styleMemoryText(style),
      db: input.db,
      defer: input.defer,
      hindsight: input.hindsight,
      logger: input.logger,
      localTargetId: style.id,
      localTargetKind: "style",
      organizationId: input.organizationId,
      persona,
      title: `${style.register} ${style.sentenceLength}`.trim(),
      userId: input.userId,
    });
  }

  // Reconsolidation pass: confirmed beliefs strengthen, contradicted beliefs
  // weaken (or get superseded by their revision), and the evidence trail is
  // preserved on the row instead of being overwritten.
  let reinforcedBeliefs = 0;
  let contradictedBeliefs = 0;
  for (const plan of beliefPlans) {
    if (plan.kind === "reinforce") {
      const next = applyBeliefReinforcement(plan.target, {
        evidenceConfidence: plan.entry.confidence,
      });
      await input.db
        .update(personaSemanticBeliefs)
        .set({
          confidence: next.confidence,
          strength: next.strength,
          supportingSourceIds: [
            ...new Set([...plan.target.supportingSourceIds, ...sourceIds]),
          ],
        })
        .where(eq(personaSemanticBeliefs.id, plan.target.id));
      reinforcedBeliefs += 1;
      continue;
    }
    if (plan.kind !== "contradict") {
      continue;
    }
    const next = applyBeliefContradiction(plan.target, {
      evidenceConfidence: plan.entry.confidence,
    });
    const superseded = isRevision(plan);
    await input.db
      .update(personaSemanticBeliefs)
      .set({
        confidence: next.confidence,
        contradictingSourceIds: [
          ...new Set([...plan.target.contradictingSourceIds, ...sourceIds]),
        ],
        state: superseded ? "superseded" : next.state,
        strength: next.strength,
        temporalValidity: superseded
          ? { ...plan.target.temporalValidity, to: nowIso }
          : plan.target.temporalValidity,
      })
      .where(eq(personaSemanticBeliefs.id, plan.target.id));
    contradictedBeliefs += 1;
  }

  await input.db
    .update(personaInteractionMemories)
    .set({ state: "consolidated" })
    .where(
      and(
        eq(personaInteractionMemories.organizationId, input.organizationId),
        eq(personaInteractionMemories.personaId, persona.id),
        or(
          ...interactions.map((interaction) =>
            eq(personaInteractionMemories.id, interaction.id)
          )
        )
      )
    );

  return {
    contradictedBeliefs,
    createdBeliefs: createdBeliefs.length,
    createdEpisodes: createdEpisodes.length,
    createdHabits: createdHabits.length,
    createdStyleProfiles: createdStyleProfiles.length,
    personaKey: persona.personaKey,
    reinforcedBeliefs,
  };
}

export async function processPersonaMemoryConsolidationTick(input: {
  consolidate: PersonaJsonLlm;
  db: Database;
  embed?: PersonaEmbedder;
  logger?: PersonaLogger;
  now?: Date;
}): Promise<{
  processedScopes: number;
  contradictedBeliefs: number;
  createdBeliefs: number;
  createdEpisodes: number;
  createdHabits: number;
  createdStyleProfiles: number;
  embeddedMemories: number;
  reinforcedBeliefs: number;
}> {
  const candidates = await input.db
    .select({
      organizationId: personaInteractionMemories.organizationId,
      personaId: personaInteractionMemories.personaId,
      personaKey: personaProfiles.personaKey,
      userId: personaInteractionMemories.userId,
    })
    .from(personaInteractionMemories)
    .innerJoin(
      personaProfiles,
      eq(personaProfiles.id, personaInteractionMemories.personaId)
    )
    .where(
      and(
        eq(personaInteractionMemories.state, "consolidation_candidate"),
        eq(personaInteractionMemories.shouldConsolidate, true)
      )
    )
    .limit(25);

  const scopeKeys = new Set<string>();
  const scopes = candidates.filter((candidate) => {
    const key = `${candidate.organizationId}:${candidate.personaKey}:${
      candidate.userId ?? ""
    }`;
    if (scopeKeys.has(key)) {
      return false;
    }
    scopeKeys.add(key);
    return true;
  });

  const totals = {
    contradictedBeliefs: 0,
    createdBeliefs: 0,
    createdEpisodes: 0,
    createdHabits: 0,
    createdStyleProfiles: 0,
    embeddedMemories: 0,
    processedScopes: 0,
    reinforcedBeliefs: 0,
  };
  for (const scope of scopes) {
    const result = await consolidatePersonaMemoryScope({
      consolidate: input.consolidate,
      db: input.db,
      embed: input.embed,
      logger: input.logger,
      organizationId: scope.organizationId,
      personaKey: scope.personaKey,
      userId: scope.userId,
    });
    totals.contradictedBeliefs += result.contradictedBeliefs;
    totals.createdBeliefs += result.createdBeliefs;
    totals.createdEpisodes += result.createdEpisodes;
    totals.createdHabits += result.createdHabits;
    totals.createdStyleProfiles += result.createdStyleProfiles;
    totals.processedScopes += 1;
    totals.reinforcedBeliefs += result.reinforcedBeliefs;
  }

  // Idle replay: each tick also embeds a bounded batch of memories that were
  // written before embeddings existed or while the provider was unavailable.
  try {
    const backfill = await backfillPersonaMemoryEmbeddings(input.db, {
      embed: input.embed,
      logger: input.logger,
    });
    totals.embeddedMemories = backfill.embedded;
  } catch (error) {
    const logger = resolvePersonaLogger(input.logger);
    logger.warn(
      personaLogMessage("[persona-memory] embedding backfill failed:", error)
    );
  }
  return totals;
}
