import { and, eq, ne } from "drizzle-orm";

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
import type { PersonaProfile } from "../schema";

export type PersonaProfileDeleteCounts = {
  aliases: number;
  beliefs: number;
  embeddings: number;
  emotionalSalience: number;
  episodes: number;
  externalMemoryRefs: number;
  facts: number;
  habits: number;
  interactionMemories: number;
  moodStates: number;
  sourceChunks: number;
  sourceDocuments: number;
  styleProfiles: number;
  workspaceStates: number;
};

export async function deletePersonaProfile(
  db: Database,
  input: {
    organizationId: string;
    personaKey: string;
    updatedByUserId?: string | null;
  }
): Promise<{
  deletedCounts: PersonaProfileDeleteCounts;
  profile: PersonaProfile;
}> {
  const personaKey = input.personaKey.trim();
  if (!/^[a-z][a-z0-9._-]{0,79}$/.test(personaKey)) {
    throw new Error(
      "personaKey must start with a lowercase letter and only contain lowercase letters, numbers, dots, underscores, or dashes."
    );
  }

  return db.transaction(async (tx) => {
    const [persona] = await tx
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
    if (!persona) {
      throw new Error(
        `Persona ${personaKey} is not configured for this organization.`
      );
    }

    const now = new Date();
    const updatedByUserId = input.updatedByUserId ?? null;
    const aliases = (
      await tx
        .update(personaAliases)
        .set({ state: "deleted", updatedAt: now, updatedByUserId })
        .where(
          and(
            eq(personaAliases.organizationId, input.organizationId),
            eq(personaAliases.personaId, persona.id),
            ne(personaAliases.state, "deleted")
          )
        )
        .returning({ id: personaAliases.id })
    ).length;
    const embeddings = (
      await tx
        .delete(personaMemoryEmbeddings)
        .where(eq(personaMemoryEmbeddings.personaId, persona.id))
        .returning({ id: personaMemoryEmbeddings.id })
    ).length;
    const emotionalSalience = (
      await tx
        .delete(personaEmotionalSalience)
        .where(eq(personaEmotionalSalience.personaId, persona.id))
        .returning({ id: personaEmotionalSalience.id })
    ).length;
    const interactionMemories = (
      await tx
        .delete(personaInteractionMemories)
        .where(eq(personaInteractionMemories.personaId, persona.id))
        .returning({ id: personaInteractionMemories.id })
    ).length;
    const moodStates = (
      await tx
        .delete(personaMoodStates)
        .where(eq(personaMoodStates.personaId, persona.id))
        .returning({ id: personaMoodStates.id })
    ).length;
    const workspaceStates = (
      await tx
        .delete(personaWorkspaceStates)
        .where(eq(personaWorkspaceStates.personaId, persona.id))
        .returning({ id: personaWorkspaceStates.id })
    ).length;
    const externalMemoryRefs = (
      await tx
        .update(personaExternalMemoryRefs)
        .set({ state: "deleted", updatedAt: now })
        .where(
          and(
            eq(personaExternalMemoryRefs.personaId, persona.id),
            ne(personaExternalMemoryRefs.state, "deleted")
          )
        )
        .returning({ id: personaExternalMemoryRefs.id })
    ).length;
    const episodes = (
      await tx
        .delete(personaEpisodeMemories)
        .where(eq(personaEpisodeMemories.personaId, persona.id))
        .returning({ id: personaEpisodeMemories.id })
    ).length;
    const beliefs = (
      await tx
        .delete(personaSemanticBeliefs)
        .where(eq(personaSemanticBeliefs.personaId, persona.id))
        .returning({ id: personaSemanticBeliefs.id })
    ).length;
    const facts = (
      await tx
        .delete(personaFacts)
        .where(eq(personaFacts.personaId, persona.id))
        .returning({ id: personaFacts.id })
    ).length;
    const habits = (
      await tx
        .delete(personaHabitPatterns)
        .where(eq(personaHabitPatterns.personaId, persona.id))
        .returning({ id: personaHabitPatterns.id })
    ).length;
    const styleProfiles = (
      await tx
        .delete(personaStyleProfiles)
        .where(eq(personaStyleProfiles.personaId, persona.id))
        .returning({ id: personaStyleProfiles.id })
    ).length;
    const sourceChunks = (
      await tx
        .delete(personaSourceChunks)
        .where(eq(personaSourceChunks.personaId, persona.id))
        .returning({ id: personaSourceChunks.id })
    ).length;
    const sourceDocuments = (
      await tx
        .delete(personaSourceDocuments)
        .where(eq(personaSourceDocuments.personaId, persona.id))
        .returning({ id: personaSourceDocuments.id })
    ).length;
    const [deletedProfile] = await tx
      .update(personaProfiles)
      .set({ state: "deleted", updatedAt: now, updatedByUserId })
      .where(eq(personaProfiles.id, persona.id))
      .returning();
    if (!deletedProfile) {
      throw new Error(`Failed to delete persona ${personaKey}.`);
    }

    return {
      deletedCounts: {
        aliases,
        beliefs,
        embeddings,
        emotionalSalience,
        episodes,
        externalMemoryRefs,
        facts,
        habits,
        interactionMemories,
        moodStates,
        sourceChunks,
        sourceDocuments,
        styleProfiles,
        workspaceStates,
      },
      profile: deletedProfile,
    };
  });
}
