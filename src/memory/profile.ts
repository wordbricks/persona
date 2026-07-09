import { and, asc, eq, inArray, isNull, ne, or, sql } from "drizzle-orm";

import type { PersonaDatabase as Database } from "../db";
import {
  personaAliases,
  personaEmotionalSalience,
  personaEpisodeMemories,
  personaExternalMemoryRefs,
  personaFacts,
  personaHabitPatterns,
  personaInteractionMemories,
  personaMemoryEmbeddings,
  personaMoodStates,
  personaProfiles,
  personaSemanticBeliefs,
  personaSourceChunks,
  personaSourceDocuments,
  personaStyleProfiles,
  personaWorkspaceStates,
} from "../schema";
import type {
  NewPersonaAlias,
  NewPersonaEmotionalSalience,
  NewPersonaEpisodeMemory,
  NewPersonaHabitPattern,
  NewPersonaProfile,
  NewPersonaSemanticBelief,
  NewPersonaSourceChunk,
  NewPersonaSourceDocument,
  NewPersonaStyleProfile,
  PersonaAlias,
  PersonaAliasSurface,
  PersonaProfile,
  PersonaScope,
} from "../schema";

function normalizePersonaKey(value: string): string {
  const personaKey = value.trim();
  if (!/^[a-z][a-z0-9._-]{0,79}$/.test(personaKey)) {
    throw new Error(
      "personaKey must start with a lowercase letter and only contain lowercase letters, numbers, dots, underscores, or dashes."
    );
  }
  return personaKey;
}

function normalizePersonaAliasKey(value: string): string {
  const aliasKey = value.trim().toLocaleLowerCase();
  if (aliasKey.length === 0 || aliasKey.length > 80 || /\s/u.test(aliasKey)) {
    throw new Error(
      "persona alias must be a non-empty token with no whitespace and at most 80 characters."
    );
  }
  return aliasKey;
}

export async function loadPersonaProfile(
  db: Database,
  input: { organizationId: string; personaKey: string }
): Promise<PersonaProfile> {
  const personaKey = normalizePersonaKey(input.personaKey);
  const [existing] = await db
    .select()
    .from(personaProfiles)
    .where(
      and(
        eq(personaProfiles.organizationId, input.organizationId),
        eq(personaProfiles.personaKey, personaKey),
        eq(personaProfiles.state, "active")
      )
    )
    .limit(1);
  if (existing) {
    return existing;
  }

  throw new Error(
    `Persona ${personaKey} is not configured for this organization.`
  );
}

export async function upsertPersonaProfile(
  db: Database,
  input: {
    consentStatus?: PersonaProfile["consentStatus"];
    displayName?: string | null;
    organizationId: string;
    personaKey: string;
    personaType?: PersonaProfile["personaType"];
    personaVersion?: string | null;
    policy?: Partial<PersonaProfile["policy"]>;
    profile?: Record<string, unknown>;
    sourceRef?: string | null;
    updatedByUserId?: string | null;
  }
): Promise<PersonaProfile> {
  const personaKey = normalizePersonaKey(input.personaKey);
  const existing = await loadPersonaProfile(db, {
    organizationId: input.organizationId,
    personaKey,
  }).catch(() => null);
  const displayName =
    input.displayName?.trim() ??
    existing?.displayName ??
    personaKey
      .split(/[-_]/u)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  const policy = {
    allowedUse: input.policy?.allowedUse ?? existing?.policy.allowedUse ?? [],
    biographicalSummary:
      input.policy?.biographicalSummary ??
      existing?.policy.biographicalSummary ??
      null,
    forbiddenUse:
      input.policy?.forbiddenUse ?? existing?.policy.forbiddenUse ?? [],
    knowledgeCutoffForPersona:
      input.policy?.knowledgeCutoffForPersona ??
      existing?.policy.knowledgeCutoffForPersona ??
      null,
    transparencyLabel:
      input.policy?.transparencyLabel ??
      existing?.policy.transparencyLabel ??
      `AI persona simulation for ${displayName}.`,
  };

  const values = {
    consentStatus:
      input.consentStatus ??
      existing?.consentStatus ??
      "fictional_or_authorized",
    displayName,
    organizationId: input.organizationId,
    personaKey,
    personaType: input.personaType ?? existing?.personaType ?? "synthetic_role",
    personaVersion:
      input.personaVersion?.trim() || existing?.personaVersion || "v1",
    policy,
    profile: input.profile ?? existing?.profile ?? {},
    sourceRef: input.sourceRef ?? existing?.sourceRef ?? null,
    state: "active" as const,
    updatedAt: new Date(),
    updatedByUserId: input.updatedByUserId ?? null,
  } satisfies NewPersonaProfile;

  const [saved] = await db
    .insert(personaProfiles)
    .values(values)
    .onConflictDoUpdate({
      set: values,
      target: [personaProfiles.organizationId, personaProfiles.personaKey],
      // idx_persona_profiles_org_key is a partial unique index (public
      // persona templates use a null organization id); ON CONFLICT must carry
      // the index predicate or Postgres cannot match the index and errors.
      targetWhere: sql`organization_id is not null`,
    })
    .returning();
  if (!saved) {
    throw new Error("Failed to upsert persona profile.");
  }
  return saved;
}

export async function upsertPersonaAlias(
  db: Database,
  input: {
    aliasKey: string;
    organizationId: string;
    personaKey: string;
    state?: PersonaAlias["state"];
    surface?: PersonaAliasSurface;
    updatedByUserId?: string | null;
  }
): Promise<PersonaAlias> {
  const aliasKey = normalizePersonaAliasKey(input.aliasKey);
  const surface = input.surface ?? "slack";
  const state = input.state ?? "active";
  const persona = await loadPersonaProfile(db, {
    organizationId: input.organizationId,
    personaKey: input.personaKey,
  });

  const directPersona = await loadPersonaProfile(db, {
    organizationId: input.organizationId,
    personaKey: aliasKey,
  }).catch(() => null);
  if (directPersona && directPersona.id !== persona.id) {
    throw new Error(
      `Persona alias ${aliasKey} conflicts with active persona key ${directPersona.personaKey}.`
    );
  }

  const [existingAlias] = await db
    .select({
      personaId: personaAliases.personaId,
      personaKey: personaProfiles.personaKey,
    })
    .from(personaAliases)
    .innerJoin(
      personaProfiles,
      eq(personaAliases.personaId, personaProfiles.id)
    )
    .where(
      and(
        eq(personaAliases.organizationId, input.organizationId),
        eq(personaAliases.surface, surface),
        eq(personaAliases.aliasKey, aliasKey),
        ne(personaAliases.state, "deleted")
      )
    )
    .limit(1);
  if (existingAlias && existingAlias.personaId !== persona.id) {
    // Comment: ON CONFLICT would otherwise move an alias between personas
    // silently, which makes Slack routing changes hard to audit.
    throw new Error(
      `Persona alias ${aliasKey} is already assigned to ${existingAlias.personaKey}.`
    );
  }

  const values = {
    aliasKey,
    organizationId: input.organizationId,
    personaId: persona.id,
    state,
    surface,
    updatedAt: new Date(),
    updatedByUserId: input.updatedByUserId ?? null,
  } satisfies NewPersonaAlias;

  const [saved] = await db
    .insert(personaAliases)
    .values({
      ...values,
      createdByUserId: input.updatedByUserId ?? null,
    })
    .onConflictDoUpdate({
      set: values,
      target: [
        personaAliases.organizationId,
        personaAliases.surface,
        personaAliases.aliasKey,
      ],
      targetWhere: sql`state <> 'deleted'`,
    })
    .returning();
  if (!saved) {
    throw new Error("Failed to upsert persona alias.");
  }
  return saved;
}

export async function listPersonaAliases(
  db: Database,
  input: {
    organizationId: string;
    personaKey?: string | null;
    surface?: PersonaAliasSurface;
  }
): Promise<Array<PersonaAlias & { personaKey: string }>> {
  const personaKey = input.personaKey
    ? normalizePersonaKey(input.personaKey)
    : null;
  const rows = await db
    .select({
      alias: personaAliases,
      personaKey: personaProfiles.personaKey,
    })
    .from(personaAliases)
    .innerJoin(
      personaProfiles,
      eq(personaAliases.personaId, personaProfiles.id)
    )
    .where(
      and(
        eq(personaAliases.organizationId, input.organizationId),
        input.surface ? eq(personaAliases.surface, input.surface) : undefined,
        personaKey ? eq(personaProfiles.personaKey, personaKey) : undefined
      )
    )
    .orderBy(asc(personaAliases.surface), asc(personaAliases.aliasKey));

  return rows.map((row) => ({ ...row.alias, personaKey: row.personaKey }));
}

export async function forgetPersonaLayerMemory(
  db: Database,
  input: {
    memoryId: string;
    organizationId: string;
    personaKey: string;
    sourceTitle?: string | null;
  }
): Promise<{
  forgottenEmbeddings: number;
  forgottenMemory: {
    id: string;
    memoryKind: "episode" | "fact" | "belief" | "habit" | "style";
    title: string;
  };
  forgottenSourceChunks: number;
  forgottenSourceDocuments: number;
}> {
  const persona = await loadPersonaProfile(db, {
    organizationId: input.organizationId,
    personaKey: input.personaKey,
  });

  return db.transaction(async (tx) => {
    const memory = await findPersonaLayerMemory(tx, {
      memoryId: input.memoryId,
      organizationId: input.organizationId,
      personaId: persona.id,
    });
    if (!memory) {
      throw new Error(
        `Persona memory ${input.memoryId} is not configured for persona ${persona.personaKey}.`
      );
    }

    const sourceDocumentIds = await findSyntheticSourceDocumentIdsForMemory(
      tx,
      {
        organizationId: input.organizationId,
        persona,
        sourceDocumentIds: memory.sourceDocumentIds,
        title: input.sourceTitle ?? memory.title,
      }
    );
    const sourceChunkIds =
      sourceDocumentIds.length > 0
        ? (
            await tx
              .select({ id: personaSourceChunks.id })
              .from(personaSourceChunks)
              .where(
                and(
                  eq(personaSourceChunks.organizationId, input.organizationId),
                  eq(personaSourceChunks.personaId, persona.id),
                  inArray(
                    personaSourceChunks.sourceDocumentId,
                    sourceDocumentIds
                  )
                )
              )
          ).map((row) => row.id)
        : [];

    const targetIds = [memory.id, ...sourceChunkIds];
    const deletedEmbeddings =
      targetIds.length > 0
        ? (
            await tx
              .delete(personaMemoryEmbeddings)
              .where(inArray(personaMemoryEmbeddings.targetId, targetIds))
              .returning({ id: personaMemoryEmbeddings.id })
          ).length
        : 0;

    const forgottenMemory = await forgetLayerMemoryByKind(tx, memory);
    const deletedSourceChunks =
      sourceChunkIds.length > 0
        ? (
            await tx
              .delete(personaSourceChunks)
              .where(inArray(personaSourceChunks.id, sourceChunkIds))
              .returning({ id: personaSourceChunks.id })
          ).length
        : 0;
    const deletedSourceDocuments =
      sourceDocumentIds.length > 0
        ? (
            await tx
              .delete(personaSourceDocuments)
              .where(inArray(personaSourceDocuments.id, sourceDocumentIds))
              .returning({ id: personaSourceDocuments.id })
          ).length
        : 0;
    const externalRefTargetIds = [
      memory.id,
      ...sourceChunkIds,
      ...sourceDocumentIds,
    ];
    if (externalRefTargetIds.length > 0) {
      await tx
        .update(personaExternalMemoryRefs)
        .set({ state: "deleted", updatedAt: new Date() })
        .where(
          and(
            eq(personaExternalMemoryRefs.organizationId, input.organizationId),
            eq(personaExternalMemoryRefs.personaId, persona.id),
            inArray(
              personaExternalMemoryRefs.localTargetId,
              externalRefTargetIds
            )
          )
        );
    }

    return {
      forgottenEmbeddings: deletedEmbeddings,
      forgottenMemory,
      forgottenSourceChunks: deletedSourceChunks,
      forgottenSourceDocuments: deletedSourceDocuments,
    };
  });
}

type PersonaLayerMemoryForgetTarget = {
  id: string;
  memoryKind: "episode" | "fact" | "belief" | "habit" | "style";
  sourceDocumentIds: string[];
  title: string;
};

async function findPersonaLayerMemory(
  db: Database,
  input: {
    memoryId: string;
    organizationId: string;
    personaId: string;
  }
): Promise<PersonaLayerMemoryForgetTarget | null> {
  const [episode] = await db
    .select({
      id: personaEpisodeMemories.id,
      sourceRefs: personaEpisodeMemories.sourceRefs,
      title: personaEpisodeMemories.title,
    })
    .from(personaEpisodeMemories)
    .where(
      and(
        eq(personaEpisodeMemories.id, input.memoryId),
        eq(personaEpisodeMemories.organizationId, input.organizationId),
        eq(personaEpisodeMemories.personaId, input.personaId)
      )
    )
    .limit(1);
  if (episode) {
    return {
      id: episode.id,
      memoryKind: "episode",
      sourceDocumentIds: episode.sourceRefs
        .map((ref) => ref.sourceDocumentId)
        .filter((id): id is string => typeof id === "string" && id.length > 0),
      title: episode.title,
    };
  }

  const [belief] = await db
    .select({
      id: personaSemanticBeliefs.id,
      proposition: personaSemanticBeliefs.proposition,
      supportingSourceIds: personaSemanticBeliefs.supportingSourceIds,
    })
    .from(personaSemanticBeliefs)
    .where(
      and(
        eq(personaSemanticBeliefs.id, input.memoryId),
        eq(personaSemanticBeliefs.organizationId, input.organizationId),
        eq(personaSemanticBeliefs.personaId, input.personaId)
      )
    )
    .limit(1);
  if (belief) {
    const title = await findFirstSourceDocumentTitle(db, {
      organizationId: input.organizationId,
      personaId: input.personaId,
      sourceDocumentIds: belief.supportingSourceIds,
    });
    return {
      id: belief.id,
      memoryKind: "belief",
      sourceDocumentIds: belief.supportingSourceIds,
      title: title ?? belief.proposition,
    };
  }

  const [fact] = await db
    .select({
      claimText: personaFacts.claimText,
      id: personaFacts.id,
      sourceDocumentId: personaFacts.sourceDocumentId,
      sourceRefs: personaFacts.sourceRefs,
    })
    .from(personaFacts)
    .where(
      and(
        eq(personaFacts.id, input.memoryId),
        eq(personaFacts.organizationId, input.organizationId),
        eq(personaFacts.personaId, input.personaId)
      )
    )
    .limit(1);
  if (fact) {
    return {
      id: fact.id,
      memoryKind: "fact",
      sourceDocumentIds: [
        fact.sourceDocumentId,
        ...fact.sourceRefs
          .map((ref) => ref.sourceDocumentId)
          .filter(
            (id): id is string => typeof id === "string" && id.length > 0
          ),
      ],
      title: fact.claimText,
    };
  }

  const [habit] = await db
    .select({
      id: personaHabitPatterns.id,
      trigger: personaHabitPatterns.trigger,
    })
    .from(personaHabitPatterns)
    .where(
      and(
        eq(personaHabitPatterns.id, input.memoryId),
        eq(personaHabitPatterns.organizationId, input.organizationId),
        eq(personaHabitPatterns.personaId, input.personaId)
      )
    )
    .limit(1);
  if (habit) {
    return {
      id: habit.id,
      memoryKind: "habit",
      sourceDocumentIds: [],
      title: habit.trigger.description,
    };
  }

  const [style] = await db
    .select({
      id: personaStyleProfiles.id,
    })
    .from(personaStyleProfiles)
    .where(
      and(
        eq(personaStyleProfiles.id, input.memoryId),
        eq(personaStyleProfiles.organizationId, input.organizationId),
        eq(personaStyleProfiles.personaId, input.personaId)
      )
    )
    .limit(1);
  if (style) {
    return {
      id: style.id,
      memoryKind: "style",
      sourceDocumentIds: [],
      title: style.id,
    };
  }

  return null;
}

async function findFirstSourceDocumentTitle(
  db: Database,
  input: {
    organizationId: string;
    personaId: string;
    sourceDocumentIds: string[];
  }
): Promise<string | null> {
  if (input.sourceDocumentIds.length === 0) {
    return null;
  }
  const [source] = await db
    .select({ title: personaSourceDocuments.title })
    .from(personaSourceDocuments)
    .where(
      and(
        eq(personaSourceDocuments.organizationId, input.organizationId),
        eq(personaSourceDocuments.personaId, input.personaId),
        inArray(personaSourceDocuments.id, input.sourceDocumentIds)
      )
    )
    .limit(1);
  return source?.title ?? null;
}

async function findSyntheticSourceDocumentIdsForMemory(
  db: Database,
  input: {
    organizationId: string;
    persona: PersonaProfile;
    sourceDocumentIds: string[];
    title: string;
  }
): Promise<string[]> {
  const sourceDocumentIds = input.sourceDocumentIds.filter(Boolean);
  const title = input.title.trim();
  if (sourceDocumentIds.length === 0 && title.length === 0) {
    return [];
  }

  const idOrTitle =
    sourceDocumentIds.length > 0 && title.length > 0
      ? or(
          inArray(personaSourceDocuments.id, sourceDocumentIds),
          eq(personaSourceDocuments.title, title)
        )
      : sourceDocumentIds.length > 0
        ? inArray(personaSourceDocuments.id, sourceDocumentIds)
        : eq(personaSourceDocuments.title, title);
  const rows = await db
    .select({ id: personaSourceDocuments.id })
    .from(personaSourceDocuments)
    .where(
      and(
        eq(personaSourceDocuments.organizationId, input.organizationId),
        eq(personaSourceDocuments.personaId, input.persona.id),
        eq(personaSourceDocuments.sourceType, "seed"),
        eq(personaSourceDocuments.sourcePriority, "synthetic"),
        sql`${personaSourceDocuments.contentHash} like 'manual:%'`,
        idOrTitle
      )
    );
  return rows.map((row) => row.id);
}

async function forgetLayerMemoryByKind(
  db: Database,
  memory: PersonaLayerMemoryForgetTarget
) {
  if (memory.memoryKind === "episode") {
    const [deleted] = await db
      .delete(personaEpisodeMemories)
      .where(eq(personaEpisodeMemories.id, memory.id))
      .returning({
        id: personaEpisodeMemories.id,
        title: personaEpisodeMemories.title,
      });
    return {
      id: deleted?.id ?? memory.id,
      memoryKind: memory.memoryKind,
      title: deleted?.title ?? memory.title,
    };
  }

  if (memory.memoryKind === "belief") {
    const [deleted] = await db
      .delete(personaSemanticBeliefs)
      .where(eq(personaSemanticBeliefs.id, memory.id))
      .returning({
        id: personaSemanticBeliefs.id,
        title: personaSemanticBeliefs.proposition,
      });
    return {
      id: deleted?.id ?? memory.id,
      memoryKind: memory.memoryKind,
      title: memory.title || deleted?.title || memory.id,
    };
  }

  if (memory.memoryKind === "fact") {
    const [deleted] = await db
      .delete(personaFacts)
      .where(eq(personaFacts.id, memory.id))
      .returning({
        id: personaFacts.id,
        title: personaFacts.claimText,
      });
    return {
      id: deleted?.id ?? memory.id,
      memoryKind: memory.memoryKind,
      title: deleted?.title ?? memory.title,
    };
  }

  if (memory.memoryKind === "habit") {
    const [deleted] = await db
      .delete(personaHabitPatterns)
      .where(eq(personaHabitPatterns.id, memory.id))
      .returning({
        id: personaHabitPatterns.id,
        trigger: personaHabitPatterns.trigger,
      });
    return {
      id: deleted?.id ?? memory.id,
      memoryKind: memory.memoryKind,
      title: deleted?.trigger.description ?? memory.title,
    };
  }

  const [deleted] = await db
    .delete(personaStyleProfiles)
    .where(eq(personaStyleProfiles.id, memory.id))
    .returning({ id: personaStyleProfiles.id });
  return {
    id: deleted?.id ?? memory.id,
    memoryKind: memory.memoryKind,
    title: memory.title,
  };
}

export async function copyPersonaProfile(
  db: Database,
  input: {
    sourcePersonaKey: string;
    targetOrganizationId: string;
    targetPersonaKey?: string | null;
    updatedByUserId?: string | null;
  }
): Promise<{
  copiedCounts: {
    beliefs: number;
    episodes: number;
    habits: number;
    sourceChunks: number;
    sourceDocuments: number;
    styleProfiles: number;
  };
  persona: PersonaProfile;
  sourcePersona: PersonaProfile;
}> {
  const sourcePersonaKey = normalizePersonaKey(input.sourcePersonaKey);
  const targetPersonaKey = normalizePersonaKey(
    input.targetPersonaKey ?? input.sourcePersonaKey
  );

  return db.transaction(async (tx) => {
    const sourcePersona = await loadPublicPersonaProfile(tx, sourcePersonaKey);
    const existingTarget = await loadPersonaProfile(tx, {
      organizationId: input.targetOrganizationId,
      personaKey: targetPersonaKey,
    }).catch(() => null);
    if (existingTarget) {
      throw new Error(
        `Persona ${targetPersonaKey} is already configured for the target organization.`
      );
    }

    const cloned = await clonePersonaIntoScope(tx, {
      personaScope: "organization",
      sourcePersona,
      targetOrganizationId: input.targetOrganizationId,
      targetPersonaKey,
      updatedByUserId: input.updatedByUserId ?? null,
    });
    return { ...cloned, sourcePersona };
  });
}

// Publishes an organization persona as a global public template (null
// organization id, persona_scope "public") so other orgs can browse and copy
// it. The inverse of copyPersonaProfile.
export async function publishPersonaProfile(
  db: Database,
  input: {
    organizationId: string;
    personaKey: string;
    publicPersonaKey?: string | null;
    updatedByUserId?: string | null;
  }
): Promise<{
  persona: PersonaProfile;
  publishedCounts: {
    beliefs: number;
    episodes: number;
    habits: number;
    sourceChunks: number;
    sourceDocuments: number;
    styleProfiles: number;
  };
  sourcePersona: PersonaProfile;
}> {
  const personaKey = normalizePersonaKey(input.personaKey);
  const publicPersonaKey = normalizePersonaKey(
    input.publicPersonaKey ?? input.personaKey
  );

  return db.transaction(async (tx) => {
    const sourcePersona = await loadPersonaProfile(tx, {
      organizationId: input.organizationId,
      personaKey,
    });
    const existingPublic = await loadPublicPersonaProfile(
      tx,
      publicPersonaKey
    ).catch(() => null);
    if (existingPublic) {
      const refreshedPersona = await refreshPublishedPersonaProfile(tx, {
        existingPublic,
        sourcePersona,
        updatedByUserId: input.updatedByUserId ?? null,
      });
      await clearPersonaMemoryLayers(tx, refreshedPersona);
      const cloned = await clonePersonaIntoScope(tx, {
        personaScope: "public",
        sourcePersona,
        targetOrganizationId: null,
        targetPersona: refreshedPersona,
        targetPersonaKey: publicPersonaKey,
        updatedByUserId: input.updatedByUserId ?? null,
      });
      return {
        persona: cloned.persona,
        publishedCounts: cloned.copiedCounts,
        sourcePersona,
      };
    }

    const cloned = await clonePersonaIntoScope(tx, {
      personaScope: "public",
      sourcePersona,
      targetOrganizationId: null,
      targetPersonaKey: publicPersonaKey,
      updatedByUserId: input.updatedByUserId ?? null,
    });
    return {
      persona: cloned.persona,
      publishedCounts: cloned.copiedCounts,
      sourcePersona,
    };
  });
}

async function refreshPublishedPersonaProfile(
  tx: Database,
  input: {
    existingPublic: PersonaProfile;
    sourcePersona: PersonaProfile;
    updatedByUserId: string | null;
  }
): Promise<PersonaProfile> {
  const [refreshedPersona] = await tx
    .update(personaProfiles)
    .set({
      consentStatus: input.sourcePersona.consentStatus,
      displayName: input.sourcePersona.displayName,
      personaType: input.sourcePersona.personaType,
      personaVersion: input.sourcePersona.personaVersion,
      policy: input.sourcePersona.policy,
      profile: input.sourcePersona.profile,
      sourceRef: input.sourcePersona.sourceRef,
      state: input.sourcePersona.state,
      updatedAt: new Date(),
      updatedByUserId: input.updatedByUserId,
    } satisfies Partial<NewPersonaProfile>)
    .where(eq(personaProfiles.id, input.existingPublic.id))
    .returning();
  if (!refreshedPersona) {
    throw new Error("Failed to refresh published persona profile.");
  }
  return refreshedPersona;
}

async function clearPersonaMemoryLayers(
  tx: Database,
  persona: PersonaProfile
): Promise<void> {
  await tx
    .delete(personaMemoryEmbeddings)
    .where(eq(personaMemoryEmbeddings.personaId, persona.id));
  await tx
    .delete(personaEmotionalSalience)
    .where(eq(personaEmotionalSalience.personaId, persona.id));
  await tx
    .delete(personaInteractionMemories)
    .where(eq(personaInteractionMemories.personaId, persona.id));
  await tx
    .delete(personaMoodStates)
    .where(eq(personaMoodStates.personaId, persona.id));
  await tx
    .delete(personaWorkspaceStates)
    .where(eq(personaWorkspaceStates.personaId, persona.id));
  await tx
    .delete(personaEpisodeMemories)
    .where(eq(personaEpisodeMemories.personaId, persona.id));
  await tx
    .delete(personaSemanticBeliefs)
    .where(eq(personaSemanticBeliefs.personaId, persona.id));
  await tx
    .delete(personaHabitPatterns)
    .where(eq(personaHabitPatterns.personaId, persona.id));
  await tx
    .delete(personaStyleProfiles)
    .where(eq(personaStyleProfiles.personaId, persona.id));
  await tx
    .delete(personaSourceChunks)
    .where(eq(personaSourceChunks.personaId, persona.id));
  await tx
    .delete(personaSourceDocuments)
    .where(eq(personaSourceDocuments.personaId, persona.id));
}

// Shared deep clone of a persona profile and all memory layers into a new
// scope, remapping cross-references. Used by copyPersonaProfile (public
// template -> org) and publishPersonaProfile (org -> public template).
async function clonePersonaIntoScope(
  tx: Database,
  input: {
    personaScope: PersonaScope;
    sourcePersona: PersonaProfile;
    targetOrganizationId: string | null;
    targetPersona?: PersonaProfile;
    targetPersonaKey: string;
    updatedByUserId: string | null;
  }
): Promise<{
  copiedCounts: {
    beliefs: number;
    episodes: number;
    habits: number;
    sourceChunks: number;
    sourceDocuments: number;
    styleProfiles: number;
  };
  persona: PersonaProfile;
}> {
  const sourcePersona = input.sourcePersona;
  const sourceOrganizationId = sourcePersona.organizationId;

  const copiedPersona =
    input.targetPersona ??
    (
      await tx
        .insert(personaProfiles)
        .values({
          consentStatus: sourcePersona.consentStatus,
          displayName: sourcePersona.displayName,
          organizationId: input.targetOrganizationId,
          personaKey: input.targetPersonaKey,
          personaScope: input.personaScope,
          personaType: sourcePersona.personaType,
          personaVersion: sourcePersona.personaVersion,
          policy: sourcePersona.policy,
          profile: sourcePersona.profile,
          sourceRef: sourcePersona.sourceRef,
          state: sourcePersona.state,
          updatedAt: new Date(),
          updatedByUserId: input.updatedByUserId ?? null,
        } satisfies NewPersonaProfile)
        .returning()
    )[0];
  if (!copiedPersona) {
    throw new Error("Failed to copy persona profile.");
  }

  const sourceDocuments = await tx
    .select()
    .from(personaSourceDocuments)
    .where(
      and(
        sourceOrganizationId === null
          ? isNull(personaSourceDocuments.organizationId)
          : eq(personaSourceDocuments.organizationId, sourceOrganizationId),
        eq(personaSourceDocuments.personaId, sourcePersona.id)
      )
    )
    .orderBy(asc(personaSourceDocuments.createdAt));
  const sourceDocumentIdMap = new Map<string, string>();
  let copiedSourceChunks = 0;

  for (const sourceDocument of sourceDocuments) {
    const [copiedDocument] = await tx
      .insert(personaSourceDocuments)
      .values({
        author: sourceDocument.author,
        consentStatus: sourceDocument.consentStatus,
        contentHash: sourceDocument.contentHash,
        createdAt: sourceDocument.createdAt,
        ingestedAt: sourceDocument.ingestedAt,
        metadata: sourceDocument.metadata,
        organizationId: input.targetOrganizationId,
        personaId: copiedPersona.id,
        privacyLevel: sourceDocument.privacyLevel,
        publicationDate: sourceDocument.publicationDate,
        reliability: sourceDocument.reliability,
        rightsStatus: sourceDocument.rightsStatus,
        sourcePriority: sourceDocument.sourcePriority,
        sourceType: sourceDocument.sourceType,
        sourceUri: sourceDocument.sourceUri,
        state: sourceDocument.state,
        title: sourceDocument.title,
        updatedAt: sourceDocument.updatedAt,
      } satisfies NewPersonaSourceDocument)
      .returning();
    if (!copiedDocument) {
      throw new Error("Failed to copy persona source document.");
    }
    sourceDocumentIdMap.set(sourceDocument.id, copiedDocument.id);

    const chunks = await tx
      .select()
      .from(personaSourceChunks)
      .where(eq(personaSourceChunks.sourceDocumentId, sourceDocument.id))
      .orderBy(asc(personaSourceChunks.createdAt));
    if (chunks.length > 0) {
      await tx.insert(personaSourceChunks).values(
        chunks.map(
          (chunk) =>
            ({
              createdAt: chunk.createdAt,
              embeddingId: null,
              emotions: chunk.emotions,
              endChar: chunk.endChar,
              entities: chunk.entities,
              organizationId: input.targetOrganizationId,
              personaId: copiedPersona.id,
              sourceDocumentId: copiedDocument.id,
              startChar: chunk.startChar,
              text: chunk.text,
              themes: chunk.themes,
              timeMentions: chunk.timeMentions,
            }) satisfies NewPersonaSourceChunk
        )
      );
      copiedSourceChunks += chunks.length;
    }
  }

  const sourceEpisodes = await tx
    .select()
    .from(personaEpisodeMemories)
    .where(
      and(
        sourceOrganizationId === null
          ? isNull(personaEpisodeMemories.organizationId)
          : eq(personaEpisodeMemories.organizationId, sourceOrganizationId),
        eq(personaEpisodeMemories.personaId, sourcePersona.id)
      )
    )
    .orderBy(asc(personaEpisodeMemories.createdAt));
  const episodeIdMap = new Map<string, string>();
  for (const episode of sourceEpisodes) {
    const [copiedEpisode] = await tx
      .insert(personaEpisodeMemories)
      .values({
        confidence: episode.confidence,
        createdAt: episode.createdAt,
        eventSummary: episode.eventSummary,
        firstPersonRecollection: episode.firstPersonRecollection,
        location: episode.location,
        organizationId: input.targetOrganizationId,
        people: episode.people,
        personaId: copiedPersona.id,
        privacyLevel: episode.privacyLevel,
        sourceRefs: episode.sourceRefs.map((ref) => ({
          ...ref,
          sourceDocumentId: ref.sourceDocumentId
            ? (sourceDocumentIdMap.get(ref.sourceDocumentId) ??
              ref.sourceDocumentId)
            : undefined,
        })),
        state: episode.state,
        themes: episode.themes,
        thoughtAnnotations: episode.thoughtAnnotations,
        time: episode.time,
        title: episode.title,
        updatedAt: episode.updatedAt,
      } satisfies NewPersonaEpisodeMemory)
      .returning();
    if (!copiedEpisode) {
      throw new Error("Failed to copy persona episode memory.");
    }
    episodeIdMap.set(episode.id, copiedEpisode.id);
  }

  await copyEmotionalSalience(tx, {
    episodeIdMap,
    sourceOrganizationId,
    sourcePersonaId: sourcePersona.id,
    targetOrganizationId: input.targetOrganizationId,
    targetPersonaId: copiedPersona.id,
  });

  const sourceBeliefs = await tx
    .select()
    .from(personaSemanticBeliefs)
    .where(
      and(
        sourceOrganizationId === null
          ? isNull(personaSemanticBeliefs.organizationId)
          : eq(personaSemanticBeliefs.organizationId, sourceOrganizationId),
        eq(personaSemanticBeliefs.personaId, sourcePersona.id)
      )
    )
    .orderBy(asc(personaSemanticBeliefs.createdAt));
  if (sourceBeliefs.length > 0) {
    await tx.insert(personaSemanticBeliefs).values(
      sourceBeliefs.map(
        (belief) =>
          ({
            beliefType: belief.beliefType,
            confidence: belief.confidence,
            contradictingSourceIds: remapIds(
              belief.contradictingSourceIds,
              sourceDocumentIdMap
            ),
            createdAt: belief.createdAt,
            domain: belief.domain,
            exceptions: belief.exceptions,
            firstPersonForm: belief.firstPersonForm,
            organizationId: input.targetOrganizationId,
            personaId: copiedPersona.id,
            privacyLevel: belief.privacyLevel,
            proposition: belief.proposition,
            stance: belief.stance,
            state: belief.state,
            strength: belief.strength,
            supportingMemoryIds: remapIds(
              belief.supportingMemoryIds,
              episodeIdMap
            ),
            supportingSourceIds: remapIds(
              belief.supportingSourceIds,
              sourceDocumentIdMap
            ),
            temporalValidity: belief.temporalValidity,
            updatedAt: belief.updatedAt,
          }) satisfies NewPersonaSemanticBelief
      )
    );
  }

  const sourceHabits = await tx
    .select()
    .from(personaHabitPatterns)
    .where(
      and(
        sourceOrganizationId === null
          ? isNull(personaHabitPatterns.organizationId)
          : eq(personaHabitPatterns.organizationId, sourceOrganizationId),
        eq(personaHabitPatterns.personaId, sourcePersona.id)
      )
    )
    .orderBy(asc(personaHabitPatterns.createdAt));
  if (sourceHabits.length > 0) {
    await tx.insert(personaHabitPatterns).values(
      sourceHabits.map(
        (habit) =>
          ({
            avoidPatterns: habit.avoidPatterns,
            confidence: habit.confidence,
            createdAt: habit.createdAt,
            defaultResponsePattern: habit.defaultResponsePattern,
            organizationId: input.targetOrganizationId,
            personaId: copiedPersona.id,
            rhetoricalMoves: habit.rhetoricalMoves,
            state: habit.state,
            strength: habit.strength,
            supportingExampleIds: remapIds(
              habit.supportingExampleIds,
              sourceDocumentIdMap
            ),
            tone: habit.tone,
            trigger: habit.trigger,
            updatedAt: habit.updatedAt,
          }) satisfies NewPersonaHabitPattern
      )
    );
  }

  const sourceStyles = await tx
    .select()
    .from(personaStyleProfiles)
    .where(
      and(
        sourceOrganizationId === null
          ? isNull(personaStyleProfiles.organizationId)
          : eq(personaStyleProfiles.organizationId, sourceOrganizationId),
        eq(personaStyleProfiles.personaId, sourcePersona.id)
      )
    )
    .orderBy(asc(personaStyleProfiles.createdAt));
  if (sourceStyles.length > 0) {
    await tx.insert(personaStyleProfiles).values(
      sourceStyles.map(
        (style) =>
          ({
            avoidPhrases: style.avoidPhrases,
            commonPhrases: style.commonPhrases,
            createdAt: style.createdAt,
            lexicalPreferences: style.lexicalPreferences,
            organizationId: input.targetOrganizationId,
            personaId: copiedPersona.id,
            preferredRhetoricalMoves: style.preferredRhetoricalMoves,
            register: style.register,
            sentenceLength: style.sentenceLength,
            state: style.state,
            toneVector: style.toneVector,
            updatedAt: style.updatedAt,
          }) satisfies NewPersonaStyleProfile
      )
    );
  }

  return {
    copiedCounts: {
      beliefs: sourceBeliefs.length,
      episodes: sourceEpisodes.length,
      habits: sourceHabits.length,
      sourceChunks: copiedSourceChunks,
      sourceDocuments: sourceDocuments.length,
      styleProfiles: sourceStyles.length,
    },
    persona: copiedPersona,
  };
}

export async function listPublicPersonaProfiles(
  db: Database
): Promise<PersonaProfile[]> {
  return db
    .select()
    .from(personaProfiles)
    .where(
      and(eq(personaProfiles.state, "active"), publicPersonaScopeCondition())
    )
    .orderBy(asc(personaProfiles.personaKey), asc(personaProfiles.createdAt));
}

async function loadPublicPersonaProfile(
  db: Database,
  personaKey: string
): Promise<PersonaProfile> {
  const candidates = await db
    .select()
    .from(personaProfiles)
    .where(
      and(
        eq(personaProfiles.personaKey, personaKey),
        eq(personaProfiles.state, "active"),
        publicPersonaScopeCondition()
      )
    )
    .orderBy(asc(personaProfiles.createdAt))
    .limit(2);

  if (candidates.length === 1 && candidates[0]) {
    return candidates[0];
  }
  if (candidates.length > 1) {
    throw new Error(
      `Multiple public personas are configured with key ${personaKey}.`
    );
  }
  throw new Error(`Public persona ${personaKey} is not configured.`);
}

function publicPersonaScopeCondition() {
  // Comment: Production public persona templates are represented by a null organization scope.
  return isNull(personaProfiles.organizationId);
}

function remapIds(values: string[], idMap: Map<string, string>): string[] {
  return values.map((value) => idMap.get(value) ?? value);
}

async function copyEmotionalSalience(
  db: Database,
  input: {
    episodeIdMap: Map<string, string>;
    sourceOrganizationId: string | null;
    sourcePersonaId: string;
    targetOrganizationId: string | null;
    targetPersonaId: string;
  }
): Promise<void> {
  if (input.episodeIdMap.size === 0) {
    return;
  }

  const sourceSalience = await db
    .select()
    .from(personaEmotionalSalience)
    .where(
      and(
        input.sourceOrganizationId === null
          ? isNull(personaEmotionalSalience.organizationId)
          : eq(
              personaEmotionalSalience.organizationId,
              input.sourceOrganizationId
            ),
        eq(personaEmotionalSalience.personaId, input.sourcePersonaId)
      )
    )
    .orderBy(asc(personaEmotionalSalience.createdAt));
  const values = sourceSalience.flatMap((salience) => {
    const episodeMemoryId = input.episodeIdMap.get(salience.episodeMemoryId);
    if (!episodeMemoryId) {
      return [];
    }
    return [
      {
        arousal: salience.arousal,
        confidence: salience.confidence,
        createdAt: salience.createdAt,
        dominance: salience.dominance,
        emotions: salience.emotions,
        episodeMemoryId,
        notes: salience.notes,
        organizationId: input.targetOrganizationId,
        personaId: input.targetPersonaId,
        retrievalBoost: salience.retrievalBoost,
        salienceScore: salience.salienceScore,
        selfRelevance: salience.selfRelevance,
        socialRelevance: salience.socialRelevance,
        updatedAt: salience.updatedAt,
        valence: salience.valence,
      } satisfies NewPersonaEmotionalSalience,
    ];
  });
  if (values.length > 0) {
    await db.insert(personaEmotionalSalience).values(values);
  }
}
