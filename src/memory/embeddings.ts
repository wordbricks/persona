import { and, eq, inArray, isNotNull, isNull, sql } from "drizzle-orm";

import type { PersonaDatabase as Database } from "../db";
import {
  personaEpisodeMemories,
  personaFacts,
  personaHabitPatterns,
  personaMemoryEmbeddings,
  personaSemanticBeliefs,
  personaSourceChunks,
  personaSourceDocuments,
  personaStyleProfiles,
} from "../schema";
import type {
  NewPersonaMemoryEmbedding,
  PersonaEmbeddingTargetKind,
} from "../schema";
import { DEFAULT_PERSONA_LOGGER, personaLogMessage } from "./logger";
import { generateEmbeddings } from "./openai-embedding";
import type { PersonaEmbedder, PersonaLogger } from "./types";

export function createOpenAiPersonaEmbedder(
  openaiApiKey?: string | null
): PersonaEmbedder | undefined {
  const apiKey = openaiApiKey?.trim();
  if (!apiKey) {
    return undefined;
  }
  return async (texts) => generateEmbeddings(texts, apiKey);
}

// FNV-1a over the embedded text; used to skip re-embedding unchanged memory.
export function hashPersonaMemoryText(text: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

// One text builder per memory layer, shared by retrieval candidates, write
// paths, and backfill so stored embeddings always match the matched text.
export function episodeMemoryText(episode: {
  eventSummary: string;
  themes: string[];
  title: string;
}): string {
  return `${episode.title} ${episode.eventSummary} ${episode.themes.join(" ")}`;
}

export function beliefMemoryText(belief: {
  domain: string;
  exceptions: string[];
  firstPersonForm: string | null;
  proposition: string;
}): string {
  return `${belief.domain} ${belief.proposition} ${
    belief.firstPersonForm ?? ""
  } ${belief.exceptions.join(" ")}`;
}

export function factMemoryText(fact: {
  claimText: string;
  evidenceSpan: string | null;
  factType: string;
  firstPersonForm: string | null;
  objectKey: string | null;
  objectName: string;
  objectType: string;
}): string {
  return `${fact.factType} ${fact.objectType} ${fact.objectName} ${
    fact.objectKey ?? ""
  } ${fact.claimText} ${fact.firstPersonForm ?? ""} ${fact.evidenceSpan ?? ""}`;
}

export function habitMemoryText(habit: {
  defaultResponsePattern: string[];
  rhetoricalMoves: string[];
  trigger: { type: string; description: string };
}): string {
  return `${habit.trigger.type} ${habit.trigger.description} ${habit.defaultResponsePattern.join(
    " "
  )} ${habit.rhetoricalMoves.join(" ")}`;
}

export function styleMemoryText(style: {
  avoidPhrases: string[];
  commonPhrases: string[];
  preferredRhetoricalMoves: string[];
  register: string;
  sentenceLength: string;
}): string {
  return `${style.register} ${style.sentenceLength} ${style.preferredRhetoricalMoves.join(
    " "
  )} ${style.commonPhrases.join(" ")} ${style.avoidPhrases.join(" ")}`;
}
export type PersonaMemoryEmbeddingEntry = {
  tenantId: string;
  personaId: string;
  targetId: string;
  targetKind: PersonaEmbeddingTargetKind;
  text: string;
};

// Hard cap on embedded text. The embedding API rejects oversized inputs
// (8191 tokens; CJK text can exceed that well below the equivalent ASCII
// length), and one oversized memory must never block a whole batch. 6000
// chars stays comfortably within budget while keeping enough signal for
// retrieval matching.
const MAX_EMBEDDING_TEXT_CHARS = 6_000;

export function normalizePersonaEmbeddingText(text: string): string {
  return text.trim().slice(0, MAX_EMBEDDING_TEXT_CHARS);
}

export async function upsertPersonaMemoryEmbeddings(
  db: Database,
  input: {
    embed: PersonaEmbedder;
    entries: PersonaMemoryEmbeddingEntry[];
    logger?: PersonaLogger;
  }
): Promise<number> {
  const entries = input.entries
    .map((entry) => ({
      ...entry,
      text: normalizePersonaEmbeddingText(entry.text),
    }))
    .filter((entry) => entry.targetId.length > 0 && entry.text.length > 0);
  if (entries.length === 0) {
    return 0;
  }

  const existing = await db
    .select({
      contentHash: personaMemoryEmbeddings.contentHash,
      targetId: personaMemoryEmbeddings.targetId,
      targetKind: personaMemoryEmbeddings.targetKind,
    })
    .from(personaMemoryEmbeddings)
    .where(
      inArray(
        personaMemoryEmbeddings.targetId,
        entries.map((entry) => entry.targetId)
      )
    );
  const existingHashByKey = new Map(
    existing.map((row) => [
      `${row.targetKind}:${row.targetId}`,
      row.contentHash,
    ])
  );

  const pending = entries
    .map((entry) => ({
      contentHash: hashPersonaMemoryText(entry.text),
      entry,
    }))
    .filter(
      ({ contentHash, entry }) =>
        existingHashByKey.get(`${entry.targetKind}:${entry.targetId}`) !==
        contentHash
    );
  if (pending.length === 0) {
    return 0;
  }
  const logger = input.logger ?? DEFAULT_PERSONA_LOGGER;

  try {
    return await embedAndUpsertEntries(db, input.embed, pending);
  } catch (error) {
    // One poisoned entry (provider rejection, constraint violation) must not
    // permanently wedge the batch - the backfill re-selects the same rows
    // every tick. Retry per entry and skip only the failing ones.
    logger.warn(
      personaLogMessage(
        "[persona-memory] batch embedding failed; retrying per entry:",
        error
      )
    );
    let upserted = 0;
    for (const item of pending) {
      try {
        upserted += await embedAndUpsertEntries(db, input.embed, [item]);
      } catch (entryError) {
        logger.warn(
          personaLogMessage(
            `[persona-memory] embedding skipped for ${item.entry.targetKind}:${item.entry.targetId}:`,
            entryError
          )
        );
      }
    }
    return upserted;
  }
}

async function embedAndUpsertEntries(
  db: Database,
  embed: PersonaEmbedder,
  pending: { contentHash: string; entry: PersonaMemoryEmbeddingEntry }[]
): Promise<number> {
  const vectors = await embed(pending.map(({ entry }) => entry.text));
  if (vectors.length !== pending.length) {
    throw new Error(
      `Persona embedder returned ${vectors.length} vectors for ${pending.length} texts.`
    );
  }

  const rows: NewPersonaMemoryEmbedding[] = pending.map(
    ({ contentHash, entry }, index) => ({
      contentHash,
      embedding: vectors[index] as number[],
      tenantId: entry.tenantId,
      personaId: entry.personaId,
      targetId: entry.targetId,
      targetKind: entry.targetKind,
    })
  );
  await db
    .insert(personaMemoryEmbeddings)
    .values(rows)
    .onConflictDoUpdate({
      set: {
        contentHash: sql`excluded.content_hash`,
        embedding: sql`excluded.embedding`,
        updatedAt: new Date(),
      },
      target: [
        personaMemoryEmbeddings.targetKind,
        personaMemoryEmbeddings.targetId,
      ],
    });
  return rows.length;
}

// Memory writes must never fail because the embedding provider is down; the
// backfill pass picks up anything missed here.
export async function tryUpsertPersonaMemoryEmbeddings(
  db: Database,
  input: {
    embed?: PersonaEmbedder;
    entries: PersonaMemoryEmbeddingEntry[];
    logger?: PersonaLogger;
  }
): Promise<void> {
  if (!input.embed || input.entries.length === 0) {
    return;
  }
  try {
    await upsertPersonaMemoryEmbeddings(db, {
      embed: input.embed,
      entries: input.entries,
      logger: input.logger,
    });
  } catch (error) {
    const logger = input.logger ?? DEFAULT_PERSONA_LOGGER;
    logger.warn(
      personaLogMessage("[persona-memory] embedding upsert failed:", error)
    );
  }
}

export async function backfillPersonaMemoryEmbeddings(
  db: Database,
  input: {
    embed?: PersonaEmbedder;
    limit?: number;
    logger?: PersonaLogger;
  }
): Promise<{ embedded: number }> {
  if (!input.embed) {
    return { embedded: 0 };
  }
  const limit = Math.max(1, input.limit ?? 40);
  const entries: PersonaMemoryEmbeddingEntry[] = [];
  const remaining = () => limit - entries.length;

  if (remaining() > 0) {
    const rows = await db
      .select({ episode: personaEpisodeMemories })
      .from(personaEpisodeMemories)
      .leftJoin(
        personaMemoryEmbeddings,
        and(
          eq(personaMemoryEmbeddings.targetId, personaEpisodeMemories.id),
          eq(personaMemoryEmbeddings.targetKind, "episode")
        )
      )
      .where(
        and(
          eq(personaEpisodeMemories.state, "active"),
          isNotNull(personaEpisodeMemories.tenantId),
          isNull(personaMemoryEmbeddings.id)
        )
      )
      .limit(remaining());
    entries.push(
      ...rows.flatMap(({ episode }) =>
        episode.tenantId === null
          ? []
          : [
              {
                tenantId: episode.tenantId,
                personaId: episode.personaId,
                targetId: episode.id,
                targetKind: "episode" as const,
                text: episodeMemoryText(episode),
              },
            ]
      )
    );
  }

  if (remaining() > 0) {
    const rows = await db
      .select({ belief: personaSemanticBeliefs })
      .from(personaSemanticBeliefs)
      .leftJoin(
        personaMemoryEmbeddings,
        and(
          eq(personaMemoryEmbeddings.targetId, personaSemanticBeliefs.id),
          eq(personaMemoryEmbeddings.targetKind, "belief")
        )
      )
      .where(
        and(
          eq(personaSemanticBeliefs.state, "active"),
          isNotNull(personaSemanticBeliefs.tenantId),
          isNull(personaMemoryEmbeddings.id)
        )
      )
      .limit(remaining());
    entries.push(
      ...rows.flatMap(({ belief }) =>
        belief.tenantId === null
          ? []
          : [
              {
                tenantId: belief.tenantId,
                personaId: belief.personaId,
                targetId: belief.id,
                targetKind: "belief" as const,
                text: beliefMemoryText(belief),
              },
            ]
      )
    );
  }

  if (remaining() > 0) {
    const rows = await db
      .select({ fact: personaFacts })
      .from(personaFacts)
      .leftJoin(
        personaMemoryEmbeddings,
        and(
          eq(personaMemoryEmbeddings.targetId, personaFacts.id),
          eq(personaMemoryEmbeddings.targetKind, "fact")
        )
      )
      .where(
        and(
          eq(personaFacts.state, "active"),
          isNotNull(personaFacts.tenantId),
          isNull(personaMemoryEmbeddings.id)
        )
      )
      .limit(remaining());
    entries.push(
      ...rows.flatMap(({ fact }) =>
        fact.tenantId === null
          ? []
          : [
              {
                tenantId: fact.tenantId,
                personaId: fact.personaId,
                targetId: fact.id,
                targetKind: "fact" as const,
                text: factMemoryText(fact),
              },
            ]
      )
    );
  }

  if (remaining() > 0) {
    const rows = await db
      .select({ habit: personaHabitPatterns })
      .from(personaHabitPatterns)
      .leftJoin(
        personaMemoryEmbeddings,
        and(
          eq(personaMemoryEmbeddings.targetId, personaHabitPatterns.id),
          eq(personaMemoryEmbeddings.targetKind, "habit")
        )
      )
      .where(
        and(
          eq(personaHabitPatterns.state, "active"),
          isNotNull(personaHabitPatterns.tenantId),
          isNull(personaMemoryEmbeddings.id)
        )
      )
      .limit(remaining());
    entries.push(
      ...rows.flatMap(({ habit }) =>
        habit.tenantId === null
          ? []
          : [
              {
                tenantId: habit.tenantId,
                personaId: habit.personaId,
                targetId: habit.id,
                targetKind: "habit" as const,
                text: habitMemoryText(habit),
              },
            ]
      )
    );
  }

  if (remaining() > 0) {
    const rows = await db
      .select({ style: personaStyleProfiles })
      .from(personaStyleProfiles)
      .leftJoin(
        personaMemoryEmbeddings,
        and(
          eq(personaMemoryEmbeddings.targetId, personaStyleProfiles.id),
          eq(personaMemoryEmbeddings.targetKind, "style")
        )
      )
      .where(
        and(
          eq(personaStyleProfiles.state, "active"),
          isNotNull(personaStyleProfiles.tenantId),
          isNull(personaMemoryEmbeddings.id)
        )
      )
      .limit(remaining());
    entries.push(
      ...rows.flatMap(({ style }) =>
        style.tenantId === null
          ? []
          : [
              {
                tenantId: style.tenantId,
                personaId: style.personaId,
                targetId: style.id,
                targetKind: "style" as const,
                text: styleMemoryText(style),
              },
            ]
      )
    );
  }

  if (remaining() > 0) {
    const rows = await db
      .select({ chunk: personaSourceChunks })
      .from(personaSourceChunks)
      .innerJoin(
        personaSourceDocuments,
        eq(personaSourceDocuments.id, personaSourceChunks.sourceDocumentId)
      )
      .leftJoin(
        personaMemoryEmbeddings,
        and(
          eq(personaMemoryEmbeddings.targetId, personaSourceChunks.id),
          eq(personaMemoryEmbeddings.targetKind, "source_chunk")
        )
      )
      .where(
        and(
          eq(personaSourceDocuments.state, "active"),
          isNotNull(personaSourceChunks.tenantId),
          isNull(personaMemoryEmbeddings.id)
        )
      )
      .limit(remaining());
    entries.push(
      ...rows.flatMap(({ chunk }) =>
        chunk.tenantId === null
          ? []
          : [
              {
                tenantId: chunk.tenantId,
                personaId: chunk.personaId,
                targetId: chunk.id,
                targetKind: "source_chunk" as const,
                text: chunk.text,
              },
            ]
      )
    );
  }

  if (entries.length === 0) {
    return { embedded: 0 };
  }
  const embedded = await upsertPersonaMemoryEmbeddings(db, {
    embed: input.embed,
    entries,
    logger: input.logger,
  });
  return { embedded };
}
