import { and, eq, sql } from "drizzle-orm";
import { z } from "zod";

import type { PersonaDatabase as Database } from "../db";
import {
  personaEpisodeMemories,
  personaExternalMemoryRefs,
  personaFacts,
  personaHabitPatterns,
  personaSemanticBeliefs,
  personaSourceChunks,
  personaSourceDocuments,
  personaStyleProfiles,
  PERSONA_BELIEF_STANCES,
  PERSONA_BELIEF_TYPES,
  PERSONA_FACT_OBJECT_TYPES,
  PERSONA_FACT_TYPES,
  PERSONA_PRIVACY_LEVELS,
} from "../schema";
import type {
  NewPersonaEpisodeMemory,
  NewPersonaExternalMemoryRef,
  NewPersonaFact,
  NewPersonaHabitPattern,
  NewPersonaSemanticBelief,
  NewPersonaSourceChunk,
  NewPersonaSourceDocument,
  NewPersonaStyleProfile,
  PersonaBeliefStance,
  PersonaBeliefType,
  PersonaEpisodeMemory,
  PersonaExternalMemoryLocalTargetKind,
  PersonaFact,
  PersonaFactObjectType,
  PersonaFactType,
  PersonaHabitPattern,
  PersonaMemoryState,
  PersonaPrivacyLevel,
  PersonaProfile,
  PersonaSemanticBelief,
  PersonaSourceDocument,
  PersonaStyleProfile,
  PersonaToneVector,
} from "../schema";
import { ulid } from "../schema/ulid";
import {
  beliefMemoryText,
  episodeMemoryText,
  factMemoryText,
  habitMemoryText,
  styleMemoryText,
  tryUpsertPersonaMemoryEmbeddings,
} from "./embeddings";
import type {
  HindsightPersonaMemoryClient,
  HindsightRetainInput,
  HindsightRetainResult,
} from "./hindsight";
import { hindsightPersonaBankId } from "./hindsight";
import { DEFAULT_PERSONA_LOGGER, personaLogMessage } from "./logger";
import { clampScore } from "./mood";
import { loadPersonaProfile } from "./profile";
import { tokenize } from "./selection";
import {
  MAX_HINDSIGHT_RETAIN_CONTENT_CHARS,
  MAX_SOURCE_EXCERPT_SUMMARY_CHARS,
} from "./types";
import type { PersonaEmbedder, PersonaLogger } from "./types";

export async function hashPersonaSourceContent(text: string): Promise<string> {
  const encoded = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  const hex = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
  return `sha256:${hex}`;
}

export type PersonaSourceTextChunk = {
  endChar: number;
  startChar: number;
  text: string;
};

const PERSONA_SOURCE_CHUNK_CHARS = 2_400;
const PERSONA_SOURCE_CHUNK_OVERLAP_CHARS = 180;

export function chunkPersonaSourceText(
  rawText: string,
  input: { maxChars?: number; overlapChars?: number } = {}
): PersonaSourceTextChunk[] {
  const text = rawText.trim();
  if (text.length === 0) {
    return [];
  }
  const maxChars = Math.max(400, input.maxChars ?? PERSONA_SOURCE_CHUNK_CHARS);
  const overlapChars = Math.max(
    0,
    Math.min(input.overlapChars ?? PERSONA_SOURCE_CHUNK_OVERLAP_CHARS, 400)
  );
  const chunks: PersonaSourceTextChunk[] = [];
  let start = 0;

  while (start < text.length) {
    let end = Math.min(text.length, start + maxChars);
    if (end < text.length) {
      const minBoundary = start + Math.floor(maxChars * 0.5);
      const paragraphBoundary = text.lastIndexOf("\n\n", end);
      const lineBoundary = text.lastIndexOf("\n", end);
      const sentenceBoundary = Math.max(
        text.lastIndexOf(". ", end),
        text.lastIndexOf("? ", end),
        text.lastIndexOf("! ", end),
        text.lastIndexOf("。", end),
        text.lastIndexOf("다.", end)
      );
      const spaceBoundary = text.lastIndexOf(" ", end);
      const boundary = [
        paragraphBoundary,
        lineBoundary,
        sentenceBoundary >= 0 ? sentenceBoundary + 1 : -1,
        spaceBoundary,
      ].find((candidate) => candidate >= minBoundary);
      if (boundary !== undefined) {
        end = boundary;
      }
    }

    const chunkText = text.slice(start, end).trim();
    if (chunkText.length > 0) {
      const leadingWhitespace = text
        .slice(start, end)
        .match(/^\s*/u)?.[0].length;
      const trailingWhitespace = text
        .slice(start, end)
        .match(/\s*$/u)?.[0].length;
      chunks.push({
        endChar: end - (trailingWhitespace ?? 0),
        startChar: start + (leadingWhitespace ?? 0),
        text: chunkText,
      });
    }

    if (end >= text.length) {
      break;
    }
    start = Math.max(end - overlapChars, start + 1);
  }

  return chunks;
}

export function formatPersonaSourceExcerptForQuery(input: {
  maxChars?: number;
  query: string;
  text: string;
}): string {
  const maxChars = input.maxChars ?? MAX_SOURCE_EXCERPT_SUMMARY_CHARS;
  const text = input.text.replace(/\s+/gu, " ").trim();
  if (text.length <= maxChars) {
    return text;
  }
  const lowerText = text.toLowerCase();
  const tokens = [...tokenize(input.query)].sort(
    (left, right) => right.length - left.length
  );
  const halfWindow = Math.floor(maxChars / 2);
  let bestMatch: { index: number; score: number; tokenLength: number } | null =
    null;
  for (const token of tokens) {
    let searchFrom = 0;
    while (searchFrom < lowerText.length) {
      const index = lowerText.indexOf(token.toLowerCase(), searchFrom);
      if (index < 0) {
        break;
      }
      const windowStart = Math.max(0, index - halfWindow);
      const windowEnd = Math.min(text.length, windowStart + maxChars);
      const windowText = lowerText.slice(windowStart, windowEnd);
      const score = tokens.reduce(
        (count, queryToken) =>
          windowText.includes(queryToken.toLowerCase()) ? count + 1 : count,
        0
      );
      if (
        bestMatch === null ||
        score > bestMatch.score ||
        (score === bestMatch.score && token.length > bestMatch.tokenLength) ||
        (score === bestMatch.score &&
          token.length === bestMatch.tokenLength &&
          index < bestMatch.index)
      ) {
        bestMatch = { index, score, tokenLength: token.length };
      }
      searchFrom = index + Math.max(1, token.length);
    }
  }
  if (bestMatch === null) {
    return `${text.slice(0, maxChars - 2).trim()} …`;
  }

  const start = Math.max(0, bestMatch.index - halfWindow);
  const end = Math.min(text.length, start + maxChars);
  const adjustedStart = Math.max(0, end - maxChars);
  const prefix = adjustedStart > 0 ? "… " : "";
  const suffix = end < text.length ? " …" : "";
  return `${prefix}${text.slice(adjustedStart, end).trim()}${suffix}`;
}

function schedulePersonaMemoryEffect(input: {
  defer?: (promise: Promise<unknown>) => void;
  label: string;
  logger?: PersonaLogger;
  promise: Promise<unknown>;
}): void {
  const logger = input.logger ?? DEFAULT_PERSONA_LOGGER;
  const handled = input.promise.catch((error: unknown) => {
    logger.warn(
      personaLogMessage(`[persona-memory] ${input.label} failed:`, error)
    );
  });
  if (input.defer) {
    input.defer(handled);
    return;
  }
  handled.then(() => undefined);
}

function hindsightRetainState(
  result: HindsightRetainResult
): NewPersonaExternalMemoryRef["state"] {
  if (result.succeeded) {
    return "active";
  }
  return result.attempted ? "failed" : "disabled";
}

async function upsertHindsightRetainReference(
  db: Database,
  input: {
    chatMessageId?: string | null;
    chatSessionId?: string | null;
    localTargetId: string;
    localTargetKind: PersonaExternalMemoryLocalTargetKind;
    metadata?: Record<string, unknown>;
    tenantId: string;
    personaId: string;
    privacyLevel: PersonaPrivacyLevel;
    result: HindsightRetainResult;
    retainInput: HindsightRetainInput;
    userId?: string | null;
  }
): Promise<void> {
  const bankId =
    input.result.bankId ??
    (input.result.attempted
      ? hindsightPersonaBankId({
          tenantId: input.tenantId,
          personaId: input.personaId,
          scope: input.retainInput.scope,
          userId: input.retainInput.userId,
        })
      : null);
  if (!bankId) {
    return;
  }
  const state = hindsightRetainState(input.result);
  const metadata = {
    ...(input.metadata ?? {}),
    ...(input.result.latencyMs === undefined
      ? {}
      : { latencyMs: input.result.latencyMs }),
    ...(input.result.operationId
      ? { operationId: input.result.operationId }
      : {}),
    ...(input.result.skippedReason
      ? { skippedReason: input.result.skippedReason }
      : {}),
    retainScope: input.retainInput.scope,
  };
  await db
    .insert(personaExternalMemoryRefs)
    .values({
      chatMessageId: input.chatMessageId ?? null,
      chatSessionId: input.chatSessionId ?? null,
      externalBankId: bankId,
      externalMemoryId: null,
      externalObservationId: null,
      lastError: input.result.error ?? null,
      localTargetId: input.localTargetId,
      localTargetKind: input.localTargetKind,
      metadata,
      tenantId: input.tenantId,
      personaId: input.personaId,
      privacyLevel: input.privacyLevel,
      provider: "hindsight",
      state,
      userId: input.userId ?? null,
    })
    .onConflictDoUpdate({
      set: {
        chatMessageId: input.chatMessageId ?? null,
        chatSessionId: input.chatSessionId ?? null,
        externalBankId: bankId,
        lastError: input.result.error ?? null,
        metadata,
        privacyLevel: input.privacyLevel,
        state,
        updatedAt: new Date(),
        userId: input.userId ?? null,
      },
      target: [
        personaExternalMemoryRefs.localTargetKind,
        personaExternalMemoryRefs.localTargetId,
        personaExternalMemoryRefs.provider,
      ],
    });
}

export function scheduleHindsightRetain(input: {
  chatMessageId?: string | null;
  chatSessionId?: string | null;
  db: Database;
  defer?: (promise: Promise<unknown>) => void;
  hindsight?: HindsightPersonaMemoryClient | null;
  logger?: PersonaLogger;
  localTargetId: string;
  localTargetKind: PersonaExternalMemoryLocalTargetKind;
  metadata?: Record<string, unknown>;
  tenantId: string;
  personaId: string;
  privacyLevel: PersonaPrivacyLevel;
  retainInput: HindsightRetainInput;
  userId?: string | null;
}): void {
  if (!input.hindsight || input.privacyLevel === "sensitive") {
    return;
  }
  schedulePersonaMemoryEffect({
    defer: input.defer,
    label: "hindsight retain",
    logger: input.logger,
    promise: input.hindsight.retain(input.retainInput).then(async (result) =>
      upsertHindsightRetainReference(input.db, {
        chatMessageId: input.chatMessageId,
        chatSessionId: input.chatSessionId,
        localTargetId: input.localTargetId,
        localTargetKind: input.localTargetKind,
        metadata: input.metadata,
        tenantId: input.tenantId,
        personaId: input.personaId,
        privacyLevel: input.privacyLevel,
        result,
        retainInput: input.retainInput,
        userId: input.userId,
      })
    ),
  });
}

function scheduleHindsightSourceDocumentRetain(input: {
  db: Database;
  defer?: (promise: Promise<unknown>) => void;
  hindsight?: HindsightPersonaMemoryClient | null;
  logger?: PersonaLogger;
  tenantId: string;
  persona: PersonaProfile;
  rawText: string;
  sourceDocument: PersonaSourceDocument;
}): void {
  if (
    input.sourceDocument.privacyLevel === "private" ||
    input.sourceDocument.privacyLevel === "sensitive"
  ) {
    return;
  }
  scheduleHindsightRetain({
    db: input.db,
    defer: input.defer,
    hindsight: input.hindsight,
    logger: input.logger,
    localTargetId: input.sourceDocument.id,
    localTargetKind: "source_document",
    metadata: {
      source: "ingestPersonaSourceDocument",
      sourcePriority: input.sourceDocument.sourcePriority,
      sourceType: input.sourceDocument.sourceType,
    },
    tenantId: input.tenantId,
    personaId: input.persona.id,
    privacyLevel: input.sourceDocument.privacyLevel,
    retainInput: {
      content: input.rawText.slice(0, MAX_HINDSIGHT_RETAIN_CONTENT_CHARS),
      context: `Persona source document: ${input.sourceDocument.title}`,
      documentId: `persona_source_document_${input.sourceDocument.id}`,
      tenantId: input.tenantId,
      personaId: input.persona.id,
      personaKey: input.persona.personaKey,
      privacyLevel: input.sourceDocument.privacyLevel,
      scope: "persona_global",
      tags: [
        "memory_kind_source_document",
        `source_type_${input.sourceDocument.sourceType}`,
      ],
      themes: [],
      timestamp: input.sourceDocument.createdAt,
      userId: null,
    },
    userId: null,
  });
}

export type PersonaSourceIngestionResult = {
  chunkCount: number;
  sourceDocument: PersonaSourceDocument;
};

export async function ingestPersonaSourceDocument(
  db: Database,
  input: {
    author?: string | null;
    consentStatus?: PersonaSourceDocument["consentStatus"] | null;
    defer?: (promise: Promise<unknown>) => void;
    embed?: PersonaEmbedder;
    hindsight?: HindsightPersonaMemoryClient | null;
    logger?: PersonaLogger;
    metadata?: Record<string, unknown>;
    tenantId: string;
    personaKey: string;
    privacyLevel?: PersonaPrivacyLevel | null;
    publicationDate?: string | null;
    rawText: string;
    reliability?: number | null;
    rightsStatus?: PersonaSourceDocument["rightsStatus"] | null;
    sourcePriority?: PersonaSourceDocument["sourcePriority"] | null;
    sourceType?: PersonaSourceDocument["sourceType"] | null;
    sourceUri?: string | null;
    title: string;
  }
): Promise<PersonaSourceIngestionResult> {
  const persona = await loadPersonaProfile(db, {
    tenantId: input.tenantId,
    personaKey: input.personaKey,
  });
  const rawText = input.rawText.trim();
  if (rawText.length === 0) {
    throw new Error("Persona source document content cannot be empty.");
  }
  const chunks = chunkPersonaSourceText(rawText);
  if (chunks.length === 0) {
    throw new Error("Persona source document content cannot be chunked.");
  }
  const contentHash = await hashPersonaSourceContent(rawText);
  const now = new Date();
  const values = {
    author: input.author?.trim() || null,
    consentStatus: input.consentStatus ?? persona.consentStatus,
    contentHash,
    metadata: {
      medium: "text",
      ...(input.metadata ?? {}),
    },
    tenantId: input.tenantId,
    personaId: persona.id,
    privacyLevel: input.privacyLevel ?? "internal",
    publicationDate: input.publicationDate?.trim() || null,
    reliability: clampScore(input.reliability ?? 0.75),
    rightsStatus: input.rightsStatus ?? "unknown",
    sourcePriority: input.sourcePriority ?? "unverified_secondary",
    sourceType: input.sourceType ?? "external_source",
    sourceUri: input.sourceUri?.trim() || null,
    title: input.title.trim(),
    updatedAt: now,
  } satisfies NewPersonaSourceDocument;

  const [sourceDocument] = await db
    .insert(personaSourceDocuments)
    .values(values)
    .onConflictDoUpdate({
      set: values,
      target: [
        personaSourceDocuments.tenantId,
        personaSourceDocuments.personaId,
        personaSourceDocuments.contentHash,
      ],
      targetWhere: sql`${personaSourceDocuments.contentHash} is not null`,
    })
    .returning();
  if (!sourceDocument) {
    throw new Error("Failed to ingest persona source document.");
  }

  const existingChunks = await db
    .select({ id: personaSourceChunks.id, text: personaSourceChunks.text })
    .from(personaSourceChunks)
    .where(eq(personaSourceChunks.sourceDocumentId, sourceDocument.id));
  if (existingChunks.length > 0) {
    await tryUpsertPersonaMemoryEmbeddings(db, {
      embed: input.embed,
      entries: existingChunks.map((chunk) => ({
        tenantId: input.tenantId,
        personaId: persona.id,
        targetId: chunk.id,
        targetKind: "source_chunk" as const,
        text: chunk.text,
      })),
      logger: input.logger,
    });
    return {
      chunkCount: existingChunks.length,
      sourceDocument,
    };
  }

  const insertedChunks = await db
    .insert(personaSourceChunks)
    .values(
      chunks.map(
        (chunk) =>
          ({
            endChar: chunk.endChar,
            tenantId: input.tenantId,
            personaId: persona.id,
            sourceDocumentId: sourceDocument.id,
            startChar: chunk.startChar,
            text: chunk.text,
          }) satisfies NewPersonaSourceChunk
      )
    )
    .returning();

  await tryUpsertPersonaMemoryEmbeddings(db, {
    embed: input.embed,
    entries: insertedChunks.map((chunk) => ({
      tenantId: input.tenantId,
      personaId: persona.id,
      targetId: chunk.id,
      targetKind: "source_chunk" as const,
      text: chunk.text,
    })),
    logger: input.logger,
  });
  scheduleHindsightSourceDocumentRetain({
    db,
    defer: input.defer,
    hindsight: input.hindsight,
    logger: input.logger,
    tenantId: input.tenantId,
    persona,
    rawText,
    sourceDocument,
  });

  return {
    chunkCount: insertedChunks.length,
    sourceDocument,
  };
}

export type PersonaSourceDraftEpisode = {
  confidence: number;
  eventSummary: string;
  firstPersonRecollection?: string | null;
  privacyLevel?: PersonaPrivacyLevel | null;
  quoteSpan?: string | null;
  sourceChunkIds?: string[];
  themes?: string[];
  thoughtAnnotations?: string[];
  time?: NewPersonaEpisodeMemory["time"];
  title: string;
};

export type PersonaSourceDraftBelief = {
  beliefType: PersonaBeliefType;
  confidence: number;
  domain: string;
  exceptions?: string[];
  firstPersonForm?: string | null;
  privacyLevel?: PersonaPrivacyLevel | null;
  proposition: string;
  sourceChunkIds?: string[];
  stance?: PersonaBeliefStance;
  strength?: number | null;
};

export type PersonaSourceDraftFact = {
  beliefType?: PersonaBeliefType;
  claimText?: string;
  confidence: number;
  domain?: string;
  evidenceSpan?: string | null;
  fact?: string;
  factType?: PersonaFactType | null;
  firstPersonForm?: string | null;
  objectKey?: string | null;
  objectName?: string | null;
  objectType?: PersonaFactObjectType | null;
  privacyLevel?: PersonaPrivacyLevel | null;
  sourceChunkIds?: string[];
  strength?: number | null;
};

export type PersonaSourceDraftHabit = {
  avoidPatterns?: string[];
  confidence: number;
  defaultResponsePattern: string[];
  rhetoricalMoves?: string[];
  sourceChunkIds?: string[];
  strength?: number | null;
  tone?: PersonaToneVector;
  trigger: { description: string; type: string };
};

export type PersonaSourceDraftStyleProfile = {
  avoidPhrases?: string[];
  commonPhrases?: string[];
  lexicalPreferences?: { avoids?: string[]; uses?: string[] };
  preferredRhetoricalMoves?: string[];
  register: string;
  sentenceLength: string;
  sourceChunkIds?: string[];
  toneVector?: PersonaToneVector;
};

export type PersonaSourceDraftMemoryInput = {
  beliefs?: PersonaSourceDraftBelief[];
  episodes?: PersonaSourceDraftEpisode[];
  habits?: PersonaSourceDraftHabit[];
  sourceFacts?: PersonaSourceDraftFact[];
  styleProfiles?: PersonaSourceDraftStyleProfile[];
};

const personaSourceDraftConfidenceSchema = z.number().min(0).max(1);
const personaSourceDraftNullableScoreSchema = z
  .number()
  .min(0)
  .max(1)
  .nullable()
  .optional();
const personaSourceDraftPrivacySchema = z
  .enum(PERSONA_PRIVACY_LEVELS)
  .nullable()
  .optional();
const personaSourceDraftToneSchema = z.record(z.string(), z.number());

/** Strict schema for JSON returned from PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT. */
export const personaSourceDraftMemorySchema = z
  .object({
    beliefs: z
      .array(
        z
          .object({
            beliefType: z.enum(PERSONA_BELIEF_TYPES),
            confidence: personaSourceDraftConfidenceSchema,
            domain: z.string().min(1),
            exceptions: z.array(z.string()).optional(),
            firstPersonForm: z.string().nullable().optional(),
            privacyLevel: personaSourceDraftPrivacySchema,
            proposition: z.string().min(1),
            sourceChunkIds: z.array(z.string().min(1)).optional(),
            stance: z.enum(PERSONA_BELIEF_STANCES).optional(),
            strength: personaSourceDraftNullableScoreSchema,
          })
          .strict()
      )
      .default([]),
    episodes: z
      .array(
        z
          .object({
            confidence: personaSourceDraftConfidenceSchema,
            eventSummary: z.string().min(1),
            firstPersonRecollection: z.string().nullable().optional(),
            privacyLevel: personaSourceDraftPrivacySchema,
            quoteSpan: z.string().nullable().optional(),
            sourceChunkIds: z.array(z.string().min(1)).optional(),
            themes: z.array(z.string()).optional(),
            thoughtAnnotations: z.array(z.string()).optional(),
            time: z
              .object({
                date: z.string().nullable().optional(),
                from: z.string().nullable().optional(),
                lifeStage: z.string().nullable().optional(),
                precision: z
                  .enum(["day", "month", "year", "range", "unknown"])
                  .optional(),
                to: z.string().nullable().optional(),
              })
              .strict()
              .optional(),
            title: z.string().min(1),
          })
          .strict()
      )
      .default([]),
    habits: z
      .array(
        z
          .object({
            avoidPatterns: z.array(z.string()).optional(),
            confidence: personaSourceDraftConfidenceSchema,
            defaultResponsePattern: z.array(z.string().min(1)).min(1),
            rhetoricalMoves: z.array(z.string()).optional(),
            sourceChunkIds: z.array(z.string().min(1)).optional(),
            strength: personaSourceDraftNullableScoreSchema,
            tone: personaSourceDraftToneSchema.optional(),
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
    sourceFacts: z
      .array(
        z
          .object({
            beliefType: z.enum(PERSONA_BELIEF_TYPES).optional(),
            claimText: z.string().min(1).optional(),
            confidence: personaSourceDraftConfidenceSchema,
            domain: z.string().min(1).optional(),
            evidenceSpan: z.string().nullable().optional(),
            fact: z.string().min(1).optional(),
            factType: z.enum(PERSONA_FACT_TYPES).nullable().optional(),
            firstPersonForm: z.string().nullable().optional(),
            objectKey: z.string().nullable().optional(),
            objectName: z.string().nullable().optional(),
            objectType: z
              .enum(PERSONA_FACT_OBJECT_TYPES)
              .nullable()
              .optional(),
            privacyLevel: personaSourceDraftPrivacySchema,
            sourceChunkIds: z.array(z.string().min(1)).optional(),
            strength: personaSourceDraftNullableScoreSchema,
          })
          .strict()
      )
      .default([]),
    styleProfiles: z
      .array(
        z
          .object({
            avoidPhrases: z.array(z.string()).optional(),
            commonPhrases: z.array(z.string()).optional(),
            lexicalPreferences: z
              .object({
                avoids: z.array(z.string()).optional(),
                uses: z.array(z.string()).optional(),
              })
              .strict()
              .optional(),
            preferredRhetoricalMoves: z.array(z.string()).optional(),
            register: z.string().min(1),
            sentenceLength: z.string().min(1),
            sourceChunkIds: z.array(z.string().min(1)).optional(),
            toneVector: personaSourceDraftToneSchema.optional(),
          })
          .strict()
      )
      .default([]),
  })
  .strict();

export type PersonaDraftMemorySummary = {
  id: string;
  memoryKind: "episode" | "fact" | "belief" | "habit" | "style";
  state: PersonaMemoryState;
  title: string;
};

export type PersonaSourceDraftMemoryResult = {
  drafts: PersonaDraftMemorySummary[];
  sourceDocumentId: string;
};

function assertSourceChunkRefs(input: {
  knownChunkIds: Set<string>;
  sourceChunkIds?: string[];
}) {
  const invalid = (input.sourceChunkIds ?? []).find(
    (id) => !input.knownChunkIds.has(id)
  );
  if (invalid) {
    throw new Error(
      `Persona memory candidate references source chunk ${invalid}, which does not belong to the source document.`
    );
  }
}

function sourceRefsForDraft(input: {
  quoteSpan?: string | null;
  sourceChunkIds?: string[];
  sourceDocumentId: string;
}) {
  const chunkIds = input.sourceChunkIds ?? [];
  if (chunkIds.length === 0) {
    return [
      {
        quoteSpan: input.quoteSpan ?? undefined,
        sourceDocumentId: input.sourceDocumentId,
      },
    ];
  }
  return chunkIds.map((sourceChunkId) => ({
    quoteSpan: input.quoteSpan ?? undefined,
    sourceChunkId,
    sourceDocumentId: input.sourceDocumentId,
  }));
}

function sourceIdsForDraft(input: {
  sourceChunkIds?: string[];
  sourceDocumentId: string;
}) {
  return [
    ...new Set([input.sourceDocumentId, ...(input.sourceChunkIds ?? [])]),
  ];
}

export function normalizePersonaFactObjectKey(value: string): string {
  return value.trim().replace(/\s+/gu, " ").toLocaleLowerCase();
}

function claimTextForDraftFact(fact: PersonaSourceDraftFact): string {
  const claimText = fact.claimText?.trim() || fact.fact?.trim();
  if (!claimText) {
    throw new Error("Persona sourceFact must include claimText or fact.");
  }
  return claimText;
}

function objectNameForDraftFact(
  fact: PersonaSourceDraftFact,
  claimText: string
): string {
  return (
    fact.objectName?.trim() ||
    fact.objectKey?.trim() ||
    fact.domain?.trim() ||
    claimText
  );
}

export async function draftPersonaMemoriesFromSourceDocument(
  db: Database,
  input: {
    draft: PersonaSourceDraftMemoryInput;
    embed?: PersonaEmbedder;
    logger?: PersonaLogger;
    tenantId: string;
    personaKey: string;
    sourceDocumentId: string;
  }
): Promise<PersonaSourceDraftMemoryResult> {
  const persona = await loadPersonaProfile(db, {
    tenantId: input.tenantId,
    personaKey: input.personaKey,
  });
  const [sourceDocument] = await db
    .select()
    .from(personaSourceDocuments)
    .where(
      and(
        eq(personaSourceDocuments.tenantId, input.tenantId),
        eq(personaSourceDocuments.personaId, persona.id),
        eq(personaSourceDocuments.id, input.sourceDocumentId),
        eq(personaSourceDocuments.state, "active")
      )
    )
    .limit(1);
  if (!sourceDocument) {
    throw new Error(
      `Persona source document ${input.sourceDocumentId} was not found.`
    );
  }

  const chunks = await db
    .select({ id: personaSourceChunks.id })
    .from(personaSourceChunks)
    .where(eq(personaSourceChunks.sourceDocumentId, input.sourceDocumentId));
  const knownChunkIds = new Set(chunks.map((chunk) => chunk.id));
  const draftBeliefs = input.draft.beliefs ?? [];
  const draftFacts = input.draft.sourceFacts ?? [];

  for (const episode of input.draft.episodes ?? []) {
    assertSourceChunkRefs({
      knownChunkIds,
      sourceChunkIds: episode.sourceChunkIds,
    });
  }
  for (const belief of draftBeliefs) {
    assertSourceChunkRefs({
      knownChunkIds,
      sourceChunkIds: belief.sourceChunkIds,
    });
  }
  for (const fact of draftFacts) {
    assertSourceChunkRefs({
      knownChunkIds,
      sourceChunkIds: fact.sourceChunkIds,
    });
  }
  for (const habit of input.draft.habits ?? []) {
    assertSourceChunkRefs({
      knownChunkIds,
      sourceChunkIds: habit.sourceChunkIds,
    });
  }
  for (const style of input.draft.styleProfiles ?? []) {
    assertSourceChunkRefs({
      knownChunkIds,
      sourceChunkIds: style.sourceChunkIds,
    });
  }

  const privacyLevel = sourceDocument.privacyLevel;
  const [episodes, beliefs, facts, habits, styleProfiles] = await Promise.all([
    input.draft.episodes && input.draft.episodes.length > 0
      ? db
          .insert(personaEpisodeMemories)
          .values(
            input.draft.episodes.map(
              (episode) =>
                ({
                  confidence: clampScore(episode.confidence),
                  eventSummary: episode.eventSummary,
                  firstPersonRecollection:
                    episode.firstPersonRecollection ?? null,
                  tenantId: input.tenantId,
                  personaId: persona.id,
                  privacyLevel: episode.privacyLevel ?? privacyLevel,
                  sourceRefs: sourceRefsForDraft({
                    quoteSpan: episode.quoteSpan,
                    sourceChunkIds: episode.sourceChunkIds,
                    sourceDocumentId: input.sourceDocumentId,
                  }),
                  state: "draft",
                  themes: episode.themes ?? [],
                  thoughtAnnotations: episode.thoughtAnnotations ?? [],
                  time: episode.time ?? {},
                  title: episode.title,
                }) satisfies NewPersonaEpisodeMemory
            )
          )
          .returning()
      : Promise.resolve([] as PersonaEpisodeMemory[]),
    draftBeliefs.length > 0
      ? db
          .insert(personaSemanticBeliefs)
          .values(
            draftBeliefs.map(
              (belief) =>
                ({
                  beliefType: belief.beliefType,
                  confidence: clampScore(belief.confidence),
                  domain: belief.domain,
                  exceptions: belief.exceptions ?? [],
                  firstPersonForm: belief.firstPersonForm ?? null,
                  tenantId: input.tenantId,
                  personaId: persona.id,
                  privacyLevel: belief.privacyLevel ?? privacyLevel,
                  proposition: belief.proposition,
                  stance: belief.stance ?? "support",
                  state: "draft",
                  strength: clampScore(belief.strength ?? belief.confidence),
                  supportingSourceIds: sourceIdsForDraft({
                    sourceChunkIds: belief.sourceChunkIds,
                    sourceDocumentId: input.sourceDocumentId,
                  }),
                }) satisfies NewPersonaSemanticBelief
            )
          )
          .returning()
      : Promise.resolve([] as PersonaSemanticBelief[]),
    draftFacts.length > 0
      ? db
          .insert(personaFacts)
          .values(
            draftFacts.map((fact) => {
              const claimText = claimTextForDraftFact(fact);
              const objectName = objectNameForDraftFact(fact, claimText);
              const sourceChunkId = fact.sourceChunkIds?.[0] ?? null;
              return {
                claimText,
                confidence: clampScore(fact.confidence),
                evidenceSpan: fact.evidenceSpan ?? null,
                factType: fact.factType ?? "other",
                firstPersonForm: fact.firstPersonForm ?? null,
                objectKey:
                  fact.objectKey?.trim() ||
                  normalizePersonaFactObjectKey(objectName),
                objectName,
                objectType: fact.objectType ?? "text",
                tenantId: input.tenantId,
                personaId: persona.id,
                privacyLevel: fact.privacyLevel ?? privacyLevel,
                sourceChunkId,
                sourceDocumentId: input.sourceDocumentId,
                sourceRefs: sourceRefsForDraft({
                  quoteSpan: fact.evidenceSpan,
                  sourceChunkIds: fact.sourceChunkIds,
                  sourceDocumentId: input.sourceDocumentId,
                }),
                state: "draft",
              } satisfies NewPersonaFact;
            })
          )
          .returning()
      : Promise.resolve([] as PersonaFact[]),
    input.draft.habits && input.draft.habits.length > 0
      ? db
          .insert(personaHabitPatterns)
          .values(
            input.draft.habits.map(
              (habit) =>
                ({
                  avoidPatterns: habit.avoidPatterns ?? [],
                  confidence: clampScore(habit.confidence),
                  defaultResponsePattern: habit.defaultResponsePattern,
                  tenantId: input.tenantId,
                  personaId: persona.id,
                  rhetoricalMoves: habit.rhetoricalMoves ?? [],
                  state: "draft",
                  strength: clampScore(habit.strength ?? habit.confidence),
                  supportingExampleIds: sourceIdsForDraft({
                    sourceChunkIds: habit.sourceChunkIds,
                    sourceDocumentId: input.sourceDocumentId,
                  }),
                  tone: habit.tone ?? {},
                  trigger: habit.trigger,
                }) satisfies NewPersonaHabitPattern
            )
          )
          .returning()
      : Promise.resolve([] as PersonaHabitPattern[]),
    input.draft.styleProfiles && input.draft.styleProfiles.length > 0
      ? db
          .insert(personaStyleProfiles)
          .values(
            input.draft.styleProfiles.map(
              (style) =>
                ({
                  avoidPhrases: style.avoidPhrases ?? [],
                  commonPhrases: style.commonPhrases ?? [],
                  lexicalPreferences: style.lexicalPreferences ?? {},
                  tenantId: input.tenantId,
                  personaId: persona.id,
                  preferredRhetoricalMoves:
                    style.preferredRhetoricalMoves ?? [],
                  register: style.register,
                  sentenceLength: style.sentenceLength,
                  state: "draft",
                  toneVector: style.toneVector ?? {},
                }) satisfies NewPersonaStyleProfile
            )
          )
          .returning()
      : Promise.resolve([] as PersonaStyleProfile[]),
  ]);

  await tryUpsertPersonaMemoryEmbeddings(db, {
    embed: input.embed,
    entries: [
      ...episodes.map((episode) => ({
        tenantId: input.tenantId,
        personaId: persona.id,
        targetId: episode.id,
        targetKind: "episode" as const,
        text: episodeMemoryText(episode),
      })),
      ...beliefs.map((belief) => ({
        tenantId: input.tenantId,
        personaId: persona.id,
        targetId: belief.id,
        targetKind: "belief" as const,
        text: beliefMemoryText(belief),
      })),
      ...facts.map((fact) => ({
        tenantId: input.tenantId,
        personaId: persona.id,
        targetId: fact.id,
        targetKind: "fact" as const,
        text: factMemoryText(fact),
      })),
      ...habits.map((habit) => ({
        tenantId: input.tenantId,
        personaId: persona.id,
        targetId: habit.id,
        targetKind: "habit" as const,
        text: habitMemoryText(habit),
      })),
      ...styleProfiles.map((style) => ({
        tenantId: input.tenantId,
        personaId: persona.id,
        targetId: style.id,
        targetKind: "style" as const,
        text: styleMemoryText(style),
      })),
    ],
    logger: input.logger,
  });

  return {
    drafts: [
      ...episodes.map((episode) => ({
        id: episode.id,
        memoryKind: "episode" as const,
        state: episode.state,
        title: episode.title,
      })),
      ...beliefs.map((belief) => ({
        id: belief.id,
        memoryKind: "belief" as const,
        state: belief.state,
        title: belief.proposition,
      })),
      ...facts.map((fact) => ({
        id: fact.id,
        memoryKind: "fact" as const,
        state: fact.state,
        title: fact.claimText,
      })),
      ...habits.map((habit) => ({
        id: habit.id,
        memoryKind: "habit" as const,
        state: habit.state,
        title: habit.trigger.description,
      })),
      ...styleProfiles.map((style) => ({
        id: style.id,
        memoryKind: "style" as const,
        state: style.state,
        title: `${style.register} ${style.sentenceLength}`.trim(),
      })),
    ],
    sourceDocumentId: input.sourceDocumentId,
  };
}

export async function activatePersonaDraftMemory(
  db: Database,
  input: {
    memoryId: string;
    memoryKind: "episode" | "fact" | "belief" | "habit" | "style";
    tenantId: string;
    personaKey: string;
  }
): Promise<PersonaDraftMemorySummary | null> {
  const persona = await loadPersonaProfile(db, {
    tenantId: input.tenantId,
    personaKey: input.personaKey,
  });
  const values = { state: "active" as const, updatedAt: new Date() };

  if (input.memoryKind === "episode") {
    const [episode] = await db
      .update(personaEpisodeMemories)
      .set(values)
      .where(
        and(
          eq(personaEpisodeMemories.tenantId, input.tenantId),
          eq(personaEpisodeMemories.personaId, persona.id),
          eq(personaEpisodeMemories.id, input.memoryId),
          eq(personaEpisodeMemories.state, "draft")
        )
      )
      .returning();
    return episode
      ? {
          id: episode.id,
          memoryKind: "episode",
          state: episode.state,
          title: episode.title,
        }
      : null;
  }

  if (input.memoryKind === "belief") {
    const [belief] = await db
      .update(personaSemanticBeliefs)
      .set(values)
      .where(
        and(
          eq(personaSemanticBeliefs.tenantId, input.tenantId),
          eq(personaSemanticBeliefs.personaId, persona.id),
          eq(personaSemanticBeliefs.id, input.memoryId),
          eq(personaSemanticBeliefs.state, "draft")
        )
      )
      .returning();
    return belief
      ? {
          id: belief.id,
          memoryKind: "belief",
          state: belief.state,
          title: belief.proposition,
        }
      : null;
  }

  if (input.memoryKind === "fact") {
    const [fact] = await db
      .update(personaFacts)
      .set(values)
      .where(
        and(
          eq(personaFacts.tenantId, input.tenantId),
          eq(personaFacts.personaId, persona.id),
          eq(personaFacts.id, input.memoryId),
          eq(personaFacts.state, "draft")
        )
      )
      .returning();
    return fact
      ? {
          id: fact.id,
          memoryKind: "fact",
          state: fact.state,
          title: fact.claimText,
        }
      : null;
  }

  if (input.memoryKind === "habit") {
    const [habit] = await db
      .update(personaHabitPatterns)
      .set(values)
      .where(
        and(
          eq(personaHabitPatterns.tenantId, input.tenantId),
          eq(personaHabitPatterns.personaId, persona.id),
          eq(personaHabitPatterns.id, input.memoryId),
          eq(personaHabitPatterns.state, "draft")
        )
      )
      .returning();
    return habit
      ? {
          id: habit.id,
          memoryKind: "habit",
          state: habit.state,
          title: habit.trigger.description,
        }
      : null;
  }

  const [style] = await db
    .update(personaStyleProfiles)
    .set(values)
    .where(
      and(
        eq(personaStyleProfiles.tenantId, input.tenantId),
        eq(personaStyleProfiles.personaId, persona.id),
        eq(personaStyleProfiles.id, input.memoryId),
        eq(personaStyleProfiles.state, "draft")
      )
    )
    .returning();
  return style
    ? {
        id: style.id,
        memoryKind: "style",
        state: style.state,
        title: `${style.register} ${style.sentenceLength}`.trim(),
      }
    : null;
}

export async function createPersonaSourceDocument(
  db: Database,
  input: {
    authorUserId?: string | null;
    embed?: PersonaEmbedder;
    logger?: PersonaLogger;
    tenantId: string;
    persona: PersonaProfile;
    privacyLevel: PersonaPrivacyLevel;
    rawText: string;
    sourceType: PersonaSourceDocument["sourceType"];
    title: string;
  }
): Promise<PersonaSourceDocument> {
  const sourceValues = {
    author: input.authorUserId ?? null,
    consentStatus: input.persona.consentStatus,
    contentHash: `manual:${ulid()}`,
    metadata: { medium: "text" },
    tenantId: input.tenantId,
    personaId: input.persona.id,
    privacyLevel: input.privacyLevel,
    reliability: 0.85,
    rightsStatus: "owned",
    sourcePriority: "synthetic",
    sourceType: input.sourceType,
    sourceUri: `persona://${input.persona.personaKey}/${Date.now()}`,
    title: input.title,
  } satisfies NewPersonaSourceDocument;
  const [source] = await db
    .insert(personaSourceDocuments)
    .values(sourceValues)
    .returning();
  if (!source) {
    throw new Error("Failed to create persona source document.");
  }
  const [chunk] = await db
    .insert(personaSourceChunks)
    .values({
      endChar: input.rawText.length,
      tenantId: input.tenantId,
      personaId: input.persona.id,
      sourceDocumentId: source.id,
      startChar: 0,
      text: input.rawText,
    })
    .returning();
  if (chunk) {
    await tryUpsertPersonaMemoryEmbeddings(db, {
      embed: input.embed,
      entries: [
        {
          tenantId: input.tenantId,
          personaId: input.persona.id,
          targetId: chunk.id,
          targetKind: "source_chunk",
          text: chunk.text,
        },
      ],
      logger: input.logger,
    });
  }
  return source;
}
