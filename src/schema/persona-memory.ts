export const PERSONA_EMBEDDING_DIMENSION = 1536;
import { sql } from "drizzle-orm";
import {
  boolean,
  check,
  index,
  integer,
  jsonb,
  real,
  text,
  timestamp,
  uniqueIndex,
  vector,
} from "drizzle-orm/pg-core";

import { pgTable } from "./table";
import { ulid } from "./ulid";

export const PERSONA_PROFILE_TYPES = [
  "simulated_character",
  "living_public_figure",
  "deceased_public_figure",
  "private_authorized_person",
  "synthetic_role",
] as const;

export type PersonaProfileType = (typeof PERSONA_PROFILE_TYPES)[number];

export const PERSONA_CONSENT_STATUSES = [
  "explicit_consent",
  "authorized",
  "fictional_or_authorized",
  "public_material_only",
  "unknown",
] as const;

export type PersonaConsentStatus = (typeof PERSONA_CONSENT_STATUSES)[number];

export const PERSONA_PROFILE_STATES = [
  "draft",
  "review_pending",
  "active",
  "archived",
  "deleted",
] as const;

export type PersonaProfileState = (typeof PERSONA_PROFILE_STATES)[number];

export const PERSONA_ALIAS_SURFACES = ["slack", "web", "cli"] as const;

export type PersonaAliasSurface = (typeof PERSONA_ALIAS_SURFACES)[number];

export const PERSONA_ALIAS_STATES = ["active", "disabled", "deleted"] as const;

export type PersonaAliasState = (typeof PERSONA_ALIAS_STATES)[number];

const PERSONA_ALIAS_SURFACES_SQL = PERSONA_ALIAS_SURFACES.map(
  (surface) => `'${surface}'`
).join(", ");

const PERSONA_ALIAS_STATES_SQL = PERSONA_ALIAS_STATES.map(
  (state) => `'${state}'`
).join(", ");

export const PERSONA_SOURCE_TYPES = [
  "interview_transcript",
  "book",
  "essay",
  "speech",
  "conversation",
  "biography",
  "profile_document",
  "seed",
  "external_source",
  "synthetic",
] as const;

export type PersonaSourceType = (typeof PERSONA_SOURCE_TYPES)[number];

export const PERSONA_SOURCE_PRIORITIES = [
  "first_person_private",
  "first_person_public",
  "authored_work",
  "direct_interview",
  "recorded_speech",
  "verified_correspondence",
  "authorized_biography",
  "third_party_biography",
  "news_article",
  "unverified_secondary",
  "synthetic",
] as const;

export type PersonaSourcePriority = (typeof PERSONA_SOURCE_PRIORITIES)[number];

export const PERSONA_RIGHTS_STATUSES = [
  "licensed",
  "owned",
  "public",
  "fair_use_reviewed",
  "unknown",
] as const;

export type PersonaRightsStatus = (typeof PERSONA_RIGHTS_STATUSES)[number];

export const PERSONA_PRIVACY_LEVELS = [
  "public",
  "internal",
  "private",
  "sensitive",
] as const;

export type PersonaPrivacyLevel = (typeof PERSONA_PRIVACY_LEVELS)[number];

export const PERSONA_MEMORY_STATES = [
  "draft",
  "active",
  "conflicted",
  "superseded",
  "archived",
  "deleted",
] as const;

export type PersonaMemoryState = (typeof PERSONA_MEMORY_STATES)[number];

export const PERSONA_BELIEF_TYPES = [
  "moral_principle",
  "epistemic_principle",
  "political_belief",
  "aesthetic_preference",
  "relationship_belief",
  "self_concept",
  "professional_norm",
  "conflict_strategy",
] as const;

export type PersonaBeliefType = (typeof PERSONA_BELIEF_TYPES)[number];

export const PERSONA_BELIEF_STANCES = [
  "support",
  "oppose",
  "ambivalent",
  "revised",
] as const;

export type PersonaBeliefStance = (typeof PERSONA_BELIEF_STANCES)[number];

export const PERSONA_FACT_TYPES = [
  "uses_product",
  "prefers_brand",
  "recommends",
  "dislikes",
  "owns_item",
  "wears_item",
  "uses_tool",
  "eats_or_drinks",
  "works_with",
  "located_at",
  "mentions_entity",
  "other",
] as const;

export type PersonaFactType = (typeof PERSONA_FACT_TYPES)[number];

export const PERSONA_FACT_OBJECT_TYPES = [
  "brand",
  "product",
  "tool",
  "clothing",
  "food",
  "place",
  "person",
  "organization",
  "work",
  "event",
  "concept",
  "text",
] as const;

export type PersonaFactObjectType = (typeof PERSONA_FACT_OBJECT_TYPES)[number];

export const PERSONA_INTERACTION_MEMORY_STATES = [
  "recorded",
  "consolidation_candidate",
  "consolidated",
  "discarded",
  "deleted",
] as const;

export type PersonaInteractionMemoryState =
  (typeof PERSONA_INTERACTION_MEMORY_STATES)[number];

export const PERSONA_LIFECYCLE_EFFECT_TYPES = [
  "extract_episode",
  "annotate_emotion",
  "extract_belief",
  "extract_habit",
  "distill_style",
  "consolidate_memory",
  "apply_correction",
  "apply_forget",
  "grounding_review",
] as const;

export type PersonaLifecycleEffectType =
  (typeof PERSONA_LIFECYCLE_EFFECT_TYPES)[number];

export const PERSONA_LIFECYCLE_EFFECT_STATUSES = [
  "pending",
  "running",
  "completed",
  "retry_scheduled",
  "failed",
  "cancelled",
] as const;

export type PersonaLifecycleEffectStatus =
  (typeof PERSONA_LIFECYCLE_EFFECT_STATUSES)[number];

export const PERSONA_EXTERNAL_MEMORY_PROVIDERS = ["hindsight"] as const;

export type PersonaExternalMemoryProvider =
  (typeof PERSONA_EXTERNAL_MEMORY_PROVIDERS)[number];

export const PERSONA_EXTERNAL_MEMORY_REF_STATES = [
  "active",
  "deleted",
  "failed",
  "disabled",
] as const;

export type PersonaExternalMemoryRefState =
  (typeof PERSONA_EXTERNAL_MEMORY_REF_STATES)[number];

export const PERSONA_EXTERNAL_MEMORY_LOCAL_TARGET_KINDS = [
  "episode",
  "fact",
  "belief",
  "habit",
  "style",
  "interaction",
  "source_document",
  "source_chunk",
] as const;

export type PersonaExternalMemoryLocalTargetKind =
  (typeof PERSONA_EXTERNAL_MEMORY_LOCAL_TARGET_KINDS)[number];

const PERSONA_EXTERNAL_MEMORY_PROVIDERS_SQL =
  PERSONA_EXTERNAL_MEMORY_PROVIDERS.map((provider) => `'${provider}'`).join(
    ", "
  );

const PERSONA_EXTERNAL_MEMORY_REF_STATES_SQL =
  PERSONA_EXTERNAL_MEMORY_REF_STATES.map((state) => `'${state}'`).join(", ");

const PERSONA_EXTERNAL_MEMORY_LOCAL_TARGET_KINDS_SQL =
  PERSONA_EXTERNAL_MEMORY_LOCAL_TARGET_KINDS.map((kind) => `'${kind}'`).join(
    ", "
  );

export type PersonaProfilePolicy = {
  allowedUse: string[];
  forbiddenUse: string[];
  knowledgeCutoffForPersona: string | null;
  transparencyLabel: string;
  biographicalSummary: string | null;
};

export type PersonaSourceMetadata = {
  language?: string;
  medium?: string;
  url?: string | null;
  [key: string]: unknown;
};

export type PersonaTimeMetadata = {
  date?: string | null;
  precision?: "day" | "month" | "year" | "range" | "unknown";
  lifeStage?: string | null;
  from?: string | null;
  to?: string | null;
};

export type PersonaLocationMetadata = {
  name: string;
  confidence: number;
};

export type PersonaPersonRef = {
  name: string;
  relationship?: string;
  confidence: number;
};

export type PersonaEmotionAnnotation = {
  emotion: string;
  intensity: number;
  evidence?: string;
};

export type PersonaSourceRef = {
  sourceDocumentId?: string;
  sourceChunkId?: string;
  quoteSpan?: string;
};

export type PersonaToneVector = {
  warmth?: number;
  formality?: number;
  humor?: number;
  directness?: number;
  humility?: number;
  firmness?: number;
  defensiveness?: number;
};

export type PersonaWorkspacePayload = {
  gateOutput: Record<string, unknown>;
  contextKeys: Record<string, unknown>;
  activeMemories: string[];
  activeFacts: string[];
  activeBeliefs: string[];
  activeHabits: string[];
  activeStyleProfiles: string[];
  affectiveState: Record<string, unknown>;
  responsePlan: Record<string, unknown>;
  appraisalSummary: Record<string, unknown>;
  externalMemory?: PersonaExternalMemoryAudit;
  memoryRetrieval?: PersonaMemoryRetrievalPayload;
  memoryReview?: {
    createdAt: string;
    memoryId?: string | null;
    memoryIntent?: string | null;
    reason: string;
    shouldRemember: boolean;
    status: "recorded" | "skipped";
    version: 1;
  };
};

export type PersonaExternalMemoryAudit = {
  provider: "hindsight";
  enabled: boolean;
  recallAttempted: boolean;
  skippedReason?: string;
  latencyMs?: number;
  selected: PersonaExternalMemoryCandidate[];
  candidates: PersonaExternalMemoryCandidate[];
  version: 1;
};

export type PersonaExternalMemoryCandidate = {
  id: string;
  kind: "external_observation" | "external_source";
  title: string;
  text: string;
  confidence: number;
  sourceConfidence?: number | null;
  createdAt?: string | null;
  updatedAt?: string | null;
  themes: string[];
  privacyLevel: PersonaPrivacyLevel;
  provenance: {
    provider: "hindsight";
    bankId: string;
    memoryId?: string | null;
    observationId?: string | null;
    sourceIds?: string[];
    citations?: string[];
  };
  score: {
    semanticSimilarity?: number;
    temporalRelevance?: number;
    graphRelevance?: number;
    providerScore?: number;
    finalScore: number;
  };
};

export type PersonaMemoryRetrievalCandidateKind =
  | "episode"
  | "fact"
  | "belief"
  | "habit"
  | "style"
  | "source";

export type PersonaMemoryRetrievalQueryKind =
  | "episodic"
  | "semantic"
  | "habit"
  | "style"
  | "source";

export type PersonaMemoryRetrievalScoreComponents = {
  affectiveSalience: number;
  emotionLabelOverlap: number;
  entityMatch: number;
  identityRelevance: number;
  lexicalSimilarity: number;
  moodCongruence: number;
  normalizedSemanticSimilarity: number;
  privacyOrPolicyRisk: number;
  recencyOrLifeStageRelevance: number;
  semanticSimilarity: number;
  sourceConfidence: number;
  temporalMatch: number;
  themeMatch: number;
};

export type PersonaMemoryRetrievalCandidateScore = {
  activationCount: number;
  activationScore: number | null;
  availability: number | null;
  baseRank: number | null;
  components: PersonaMemoryRetrievalScoreComponents | null;
  confidence: number;
  createdAt: string;
  excludedReason: "privacy" | null;
  id: string;
  kind: PersonaMemoryRetrievalCandidateKind;
  lastActivatedAt: string | null;
  privacyLevel: PersonaPrivacyLevel;
  query: string;
  rank: number;
  rankBeforeSpreading: number | null;
  retrievalBoost: number;
  selected: boolean;
  selectedOrder: number | null;
  semanticSimilarity: number | null;
  sourceConfidence: number;
  spreadingBoost: number;
  strength: number;
};

export type PersonaMemoryRetrievalPayload = {
  budgets: Record<PersonaMemoryRetrievalCandidateKind, number>;
  candidateCount: number;
  candidates: PersonaMemoryRetrievalCandidateScore[];
  generatedAt: string;
  queries: Record<PersonaMemoryRetrievalQueryKind, string>;
  selectedCount: number;
  version: 1;
};

export const PERSONA_SCOPES = ["tenant", "public"] as const;

export type PersonaScope = (typeof PERSONA_SCOPES)[number];

// Public persona templates (persona_scope = "public") live with a null
// tenant id and are copied into tenants on demand; tenant-scoped
// rows always carry a tenant id.
export const personaProfiles = pgTable(
  "persona_profiles",
  {
    consentStatus: text("consent_status")
      .notNull()
      .$type<PersonaConsentStatus>(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    displayName: text("display_name").notNull(),
    id: text("id").primaryKey().$defaultFn(ulid),
    tenantId: text("tenant_id"),
    personaKey: text("persona_key").notNull(),
    personaScope: text("persona_scope")
      .notNull()
      .$type<PersonaScope>()
      .default("tenant"),
    personaType: text("persona_type").notNull().$type<PersonaProfileType>(),
    personaVersion: text("persona_version").notNull().default("v1"),
    policy: jsonb("policy").$type<PersonaProfilePolicy>().notNull(),
    profile: jsonb("profile").$type<Record<string, unknown>>().notNull(),
    sourceRef: text("source_ref"),
    state: text("state")
      .notNull()
      .$type<PersonaProfileState>()
      .default("active"),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
    updatedByUserId: text("updated_by_user_id"),
  },
  (table) => [
    // Partial: public persona templates have a null tenant id; tenant
    // uniqueness only applies to tenant-scoped rows. Writers using ON CONFLICT
    // against this index must pass the same predicate via targetWhere or
    // Postgres cannot match the index.
    uniqueIndex("idx_persona_profiles_tenant_key")
      .on(table.tenantId, table.personaKey)
      .where(sql`tenant_id is not null`),
    uniqueIndex("idx_persona_profiles_public_key")
      .on(table.personaKey)
      .where(sql`tenant_id is null`),
    index("idx_persona_profiles_tenant_state").on(
      table.tenantId,
      table.state
    ),
    index("idx_persona_profiles_scope_state").on(
      table.personaScope,
      table.state
    ),
  ]
);

export const personaAliases = pgTable(
  "persona_aliases",
  {
    aliasKey: text("alias_key").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    createdByUserId: text("created_by_user_id"),
    id: text("id").primaryKey().$defaultFn(ulid),
    tenantId: text("tenant_id").notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    state: text("state").notNull().$type<PersonaAliasState>().default("active"),
    surface: text("surface")
      .notNull()
      .$type<PersonaAliasSurface>()
      .default("slack"),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
    updatedByUserId: text("updated_by_user_id"),
  },
  (table) => [
    index("idx_persona_aliases_persona").on(table.personaId),
    index("idx_persona_aliases_tenant_surface_state").on(
      table.tenantId,
      table.surface,
      table.state
    ),
    uniqueIndex("idx_persona_aliases_tenant_surface_alias")
      .on(table.tenantId, table.surface, table.aliasKey)
      .where(sql`state <> 'deleted'`),
    check(
      "persona_aliases_alias_key_length_check",
      sql`char_length(${table.aliasKey}) between 1 and 80`
    ),
    check(
      "persona_aliases_alias_key_no_whitespace_check",
      sql`${table.aliasKey} !~ '\\s'`
    ),
    check(
      "persona_aliases_surface_check",
      sql`${table.surface} in (${sql.raw(PERSONA_ALIAS_SURFACES_SQL)})`
    ),
    check(
      "persona_aliases_state_check",
      sql`${table.state} in (${sql.raw(PERSONA_ALIAS_STATES_SQL)})`
    ),
  ]
);

export const personaSourceDocuments = pgTable(
  "persona_source_documents",
  {
    author: text("author"),
    consentStatus: text("consent_status")
      .notNull()
      .$type<PersonaConsentStatus>(),
    contentHash: text("content_hash"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    id: text("id").primaryKey().$defaultFn(ulid),
    ingestedAt: timestamp("ingested_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    metadata: jsonb("metadata")
      .$type<PersonaSourceMetadata>()
      .notNull()
      .default({}),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    privacyLevel: text("privacy_level")
      .notNull()
      .$type<PersonaPrivacyLevel>()
      .default("internal"),
    publicationDate: text("publication_date"),
    reliability: real("reliability").notNull().default(0.75),
    rightsStatus: text("rights_status").notNull().$type<PersonaRightsStatus>(),
    sourcePriority: text("source_priority")
      .notNull()
      .$type<PersonaSourcePriority>(),
    sourceType: text("source_type").notNull().$type<PersonaSourceType>(),
    sourceUri: text("source_uri"),
    state: text("state")
      .notNull()
      .$type<PersonaMemoryState>()
      .default("active"),
    title: text("title").notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("idx_persona_source_documents_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
    uniqueIndex("idx_persona_source_documents_hash")
      .on(table.tenantId, table.personaId, table.contentHash)
      .where(sql`${table.contentHash} is not null`),
  ]
);

export const personaSourceChunks = pgTable(
  "persona_source_chunks",
  {
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    embeddingId: text("embedding_id"),
    emotions: jsonb("emotions").$type<string[]>().notNull().default([]),
    endChar: integer("end_char"),
    entities: jsonb("entities").$type<string[]>().notNull().default([]),
    id: text("id").primaryKey().$defaultFn(ulid),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    sourceDocumentId: text("source_document_id")
      .notNull()
      .references(() => personaSourceDocuments.id, { onDelete: "cascade" }),
    startChar: integer("start_char"),
    text: text("text").notNull(),
    themes: jsonb("themes").$type<string[]>().notNull().default([]),
    timeMentions: jsonb("time_mentions")
      .$type<string[]>()
      .notNull()
      .default([]),
  },
  (table) => [
    index("idx_persona_source_chunks_document").on(table.sourceDocumentId),
    index("idx_persona_source_chunks_persona").on(
      table.tenantId,
      table.personaId
    ),
  ]
);

export const personaFacts = pgTable(
  "persona_facts",
  {
    activationCount: integer("activation_count").notNull().default(0),
    claimText: text("claim_text").notNull(),
    confidence: real("confidence").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    evidenceSpan: text("evidence_span"),
    factType: text("fact_type").notNull().$type<PersonaFactType>(),
    firstPersonForm: text("first_person_form"),
    id: text("id").primaryKey().$defaultFn(ulid),
    lastActivatedAt: timestamp("last_activated_at", { withTimezone: true }),
    objectKey: text("object_key"),
    objectName: text("object_name").notNull(),
    objectType: text("object_type").notNull().$type<PersonaFactObjectType>(),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    privacyLevel: text("privacy_level")
      .notNull()
      .$type<PersonaPrivacyLevel>()
      .default("internal"),
    sourceChunkId: text("source_chunk_id").references(
      () => personaSourceChunks.id,
      { onDelete: "set null" }
    ),
    sourceDocumentId: text("source_document_id")
      .notNull()
      .references(() => personaSourceDocuments.id, { onDelete: "cascade" }),
    sourceRefs: jsonb("source_refs")
      .$type<PersonaSourceRef[]>()
      .notNull()
      .default([]),
    state: text("state")
      .notNull()
      .$type<PersonaMemoryState>()
      .default("active"),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("idx_persona_facts_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
    index("idx_persona_facts_object").on(
      table.personaId,
      table.objectType,
      table.objectKey
    ),
    index("idx_persona_facts_source").on(
      table.sourceDocumentId,
      table.sourceChunkId
    ),
  ]
);

export const personaEpisodeMemories = pgTable(
  "persona_episode_memories",
  {
    activationCount: integer("activation_count").notNull().default(0),
    confidence: real("confidence").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    eventSummary: text("event_summary").notNull(),
    firstPersonRecollection: text("first_person_recollection"),
    id: text("id").primaryKey().$defaultFn(ulid),
    lastActivatedAt: timestamp("last_activated_at", { withTimezone: true }),
    location: jsonb("location").$type<PersonaLocationMetadata>(),
    tenantId: text("tenant_id"),
    people: jsonb("people").$type<PersonaPersonRef[]>().notNull().default([]),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    privacyLevel: text("privacy_level")
      .notNull()
      .$type<PersonaPrivacyLevel>()
      .default("internal"),
    sourceRefs: jsonb("source_refs")
      .$type<PersonaSourceRef[]>()
      .notNull()
      .default([]),
    state: text("state")
      .notNull()
      .$type<PersonaMemoryState>()
      .default("active"),
    themes: jsonb("themes").$type<string[]>().notNull().default([]),
    thoughtAnnotations: jsonb("thought_annotations")
      .$type<string[]>()
      .notNull()
      .default([]),
    time: jsonb("time").$type<PersonaTimeMetadata>().notNull().default({}),
    title: text("title").notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("idx_persona_episode_memories_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
    index("idx_persona_episode_memories_created").on(table.createdAt),
  ]
);

export const personaEmotionalSalience = pgTable(
  "persona_emotional_salience",
  {
    arousal: real("arousal").notNull(),
    confidence: real("confidence").notNull().default(0.75),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    dominance: real("dominance").notNull(),
    emotions: jsonb("emotions")
      .$type<PersonaEmotionAnnotation[]>()
      .notNull()
      .default([]),
    episodeMemoryId: text("episode_memory_id")
      .notNull()
      .references(() => personaEpisodeMemories.id, { onDelete: "cascade" }),
    id: text("id").primaryKey().$defaultFn(ulid),
    notes: text("notes"),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    retrievalBoost: real("retrieval_boost").notNull().default(1),
    salienceScore: real("salience_score").notNull(),
    selfRelevance: real("self_relevance").notNull(),
    socialRelevance: real("social_relevance").notNull().default(0),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
    valence: real("valence").notNull(),
  },
  (table) => [
    uniqueIndex("idx_persona_emotional_salience_episode").on(
      table.episodeMemoryId
    ),
    index("idx_persona_emotional_salience_persona").on(
      table.tenantId,
      table.personaId
    ),
  ]
);

// Slow-moving PAD (pleasure/valence, arousal, dominance) mood per
// (persona, user) relationship. Updated every turn with inertia and decays
// toward the persona baseline between sessions; biases memory retrieval
// (mood-congruent recall) and response tone.
export const personaMoodStates = pgTable(
  "persona_mood_states",
  {
    arousal: real("arousal").notNull().default(0.3),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    dominance: real("dominance").notNull().default(0.5),
    id: text("id").primaryKey().$defaultFn(ulid),
    tenantId: text("tenant_id").notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    turnCount: integer("turn_count").notNull().default(0),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
    userId: text("user_id").notNull(),
    valence: real("valence").notNull().default(0.1),
  },
  (table) => [
    uniqueIndex("idx_persona_mood_states_scope").on(
      table.tenantId,
      table.personaId,
      table.userId
    ),
  ]
);

export const personaSemanticBeliefs = pgTable(
  "persona_semantic_beliefs",
  {
    activationCount: integer("activation_count").notNull().default(0),
    beliefType: text("belief_type").notNull().$type<PersonaBeliefType>(),
    confidence: real("confidence").notNull(),
    contradictingSourceIds: jsonb("contradicting_source_ids")
      .$type<string[]>()
      .notNull()
      .default([]),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    domain: text("domain").notNull(),
    exceptions: jsonb("exceptions").$type<string[]>().notNull().default([]),
    firstPersonForm: text("first_person_form"),
    id: text("id").primaryKey().$defaultFn(ulid),
    lastActivatedAt: timestamp("last_activated_at", { withTimezone: true }),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    privacyLevel: text("privacy_level")
      .notNull()
      .$type<PersonaPrivacyLevel>()
      .default("internal"),
    proposition: text("proposition").notNull(),
    stance: text("stance").notNull().$type<PersonaBeliefStance>(),
    state: text("state")
      .notNull()
      .$type<PersonaMemoryState>()
      .default("active"),
    strength: real("strength").notNull(),
    supportingMemoryIds: jsonb("supporting_memory_ids")
      .$type<string[]>()
      .notNull()
      .default([]),
    supportingSourceIds: jsonb("supporting_source_ids")
      .$type<string[]>()
      .notNull()
      .default([]),
    temporalValidity: jsonb("temporal_validity")
      .$type<{ from?: string | null; to?: string | null }>()
      .notNull()
      .default({}),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("idx_persona_semantic_beliefs_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
    index("idx_persona_semantic_beliefs_domain").on(table.domain),
  ]
);

export const personaHabitPatterns = pgTable(
  "persona_habit_patterns",
  {
    activationCount: integer("activation_count").notNull().default(0),
    avoidPatterns: jsonb("avoid_patterns")
      .$type<string[]>()
      .notNull()
      .default([]),
    confidence: real("confidence").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    defaultResponsePattern: jsonb("default_response_pattern")
      .$type<string[]>()
      .notNull(),
    id: text("id").primaryKey().$defaultFn(ulid),
    lastActivatedAt: timestamp("last_activated_at", { withTimezone: true }),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    rhetoricalMoves: jsonb("rhetorical_moves")
      .$type<string[]>()
      .notNull()
      .default([]),
    state: text("state")
      .notNull()
      .$type<PersonaMemoryState>()
      .default("active"),
    strength: real("strength").notNull(),
    supportingExampleIds: jsonb("supporting_example_ids")
      .$type<string[]>()
      .notNull()
      .default([]),
    tone: jsonb("tone").$type<PersonaToneVector>().notNull().default({}),
    trigger: jsonb("trigger")
      .$type<{ type: string; description: string }>()
      .notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("idx_persona_habit_patterns_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
  ]
);

export const personaStyleProfiles = pgTable(
  "persona_style_profiles",
  {
    activationCount: integer("activation_count").notNull().default(0),
    avoidPhrases: jsonb("avoid_phrases")
      .$type<string[]>()
      .notNull()
      .default([]),
    commonPhrases: jsonb("common_phrases")
      .$type<string[]>()
      .notNull()
      .default([]),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    id: text("id").primaryKey().$defaultFn(ulid),
    lastActivatedAt: timestamp("last_activated_at", { withTimezone: true }),
    lexicalPreferences: jsonb("lexical_preferences")
      .$type<{ uses?: string[]; avoids?: string[] }>()
      .notNull()
      .default({}),
    tenantId: text("tenant_id"),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    preferredRhetoricalMoves: jsonb("preferred_rhetorical_moves")
      .$type<string[]>()
      .notNull()
      .default([]),
    register: text("register").notNull(),
    sentenceLength: text("sentence_length").notNull(),
    state: text("state")
      .notNull()
      .$type<PersonaMemoryState>()
      .default("active"),
    toneVector: jsonb("tone_vector")
      .$type<PersonaToneVector>()
      .notNull()
      .default({}),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("idx_persona_style_profiles_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
  ]
);

export const personaWorkspaceStates = pgTable(
  "persona_workspace_states",
  {
    chatMessageId: text("chat_message_id"),
    chatSessionId: text("chat_session_id"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    id: text("id").primaryKey().$defaultFn(ulid),
    tenantId: text("tenant_id").notNull(),
    payload: jsonb("payload").$type<PersonaWorkspacePayload>().notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    userId: text("user_id"),
    userMessage: text("user_message").notNull(),
  },
  (table) => [
    index("idx_persona_workspace_states_session").on(table.chatSessionId),
    index("idx_persona_workspace_states_persona").on(
      table.tenantId,
      table.personaId
    ),
  ]
);

export const personaInteractionMemories = pgTable(
  "persona_interaction_memories",
  {
    chatMessageId: text("chat_message_id"),
    chatSessionId: text("chat_session_id"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    id: text("id").primaryKey().$defaultFn(ulid),
    interactionSummary: text("interaction_summary").notNull(),
    newPersonaMemory:
      jsonb("new_persona_memory").$type<Record<string, unknown>>(),
    newUserPreference: text("new_user_preference"),
    tenantId: text("tenant_id").notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    shouldConsolidate: boolean("should_consolidate").notNull().default(false),
    state: text("state")
      .notNull()
      .$type<PersonaInteractionMemoryState>()
      .default("recorded"),
    ttlDays: integer("ttl_days").notNull().default(30),
    userId: text("user_id"),
  },
  (table) => [
    index("idx_persona_interaction_memories_persona").on(
      table.tenantId,
      table.personaId,
      table.state
    ),
    index("idx_persona_interaction_memories_session").on(table.chatSessionId),
  ]
);

export const personaExternalMemoryRefs = pgTable(
  "persona_external_memory_refs",
  {
    chatMessageId: text("chat_message_id"),
    chatSessionId: text("chat_session_id"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    externalBankId: text("external_bank_id").notNull(),
    externalMemoryId: text("external_memory_id"),
    externalObservationId: text("external_observation_id"),
    id: text("id").primaryKey().$defaultFn(ulid),
    lastError: text("last_error"),
    localTargetId: text("local_target_id"),
    localTargetKind:
      text("local_target_kind").$type<PersonaExternalMemoryLocalTargetKind>(),
    metadata: jsonb("metadata")
      .$type<Record<string, unknown>>()
      .notNull()
      .default({}),
    tenantId: text("tenant_id").notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    privacyLevel: text("privacy_level").notNull().$type<PersonaPrivacyLevel>(),
    provider: text("provider")
      .notNull()
      .$type<PersonaExternalMemoryProvider>()
      .default("hindsight"),
    state: text("state")
      .notNull()
      .$type<PersonaExternalMemoryRefState>()
      .default("active"),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
    userId: text("user_id"),
  },
  (table) => [
    index("idx_persona_external_memory_refs_persona_provider").on(
      table.tenantId,
      table.personaId,
      table.provider
    ),
    index("idx_persona_external_memory_refs_user_provider").on(
      table.tenantId,
      table.personaId,
      table.userId,
      table.provider
    ),
    index("idx_persona_external_memory_refs_external").on(
      table.provider,
      table.externalBankId,
      table.externalMemoryId
    ),
    index("idx_persona_external_memory_refs_local").on(
      table.localTargetKind,
      table.localTargetId,
      table.provider
    ),
    uniqueIndex("idx_persona_external_memory_refs_local_unique").on(
      table.localTargetKind,
      table.localTargetId,
      table.provider
    ),
    check(
      "persona_external_memory_refs_provider_check",
      sql`${table.provider} in (${sql.raw(PERSONA_EXTERNAL_MEMORY_PROVIDERS_SQL)})`
    ),
    check(
      "persona_external_memory_refs_state_check",
      sql`${table.state} in (${sql.raw(PERSONA_EXTERNAL_MEMORY_REF_STATES_SQL)})`
    ),
    check(
      "persona_external_memory_refs_local_target_kind_check",
      sql`${table.localTargetKind} is null or ${table.localTargetKind} in (${sql.raw(PERSONA_EXTERNAL_MEMORY_LOCAL_TARGET_KINDS_SQL)})`
    ),
  ]
);

export const personaLifecycleEffects = pgTable(
  "persona_lifecycle_effects",
  {
    attempt: integer("attempt").notNull().default(0),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    effectType: text("effect_type")
      .notNull()
      .$type<PersonaLifecycleEffectType>(),
    id: text("id").primaryKey().$defaultFn(ulid),
    idempotencyKey: text("idempotency_key").notNull(),
    lastError: text("last_error"),
    maxAttempts: integer("max_attempts").notNull().default(3),
    nextRunAt: timestamp("next_run_at", { withTimezone: true }),
    operationId: text("operation_id").notNull(),
    tenantId: text("tenant_id").notNull(),
    payload: jsonb("payload").$type<Record<string, unknown>>().notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    status: text("status")
      .notNull()
      .$type<PersonaLifecycleEffectStatus>()
      .default("pending"),
    targetId: text("target_id").notNull(),
    targetKind: text("target_kind").notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
    userId: text("user_id"),
  },
  (table) => [
    uniqueIndex("idx_persona_lifecycle_effects_operation").on(
      table.operationId
    ),
    uniqueIndex("idx_persona_lifecycle_effects_idempotency").on(
      table.idempotencyKey
    ),
    index("idx_persona_lifecycle_effects_status_due").on(
      table.status,
      table.nextRunAt
    ),
  ]
);

export const PERSONA_EMBEDDING_TARGET_KINDS = [
  "episode",
  "fact",
  "belief",
  "habit",
  "style",
  "source_chunk",
] as const;

export type PersonaEmbeddingTargetKind =
  (typeof PERSONA_EMBEDDING_TARGET_KINDS)[number];

// Side table instead of vector columns on each memory layer: one migration,
// uniform backfill, and retrieval can compute similarity scalars via a single
// left join per layer without shipping vectors to the app.
export const personaMemoryEmbeddings = pgTable(
  "persona_memory_embeddings",
  {
    contentHash: text("content_hash").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    embedding: vector("embedding", {
      dimensions: PERSONA_EMBEDDING_DIMENSION,
    }).notNull(),
    id: text("id").primaryKey().$defaultFn(ulid),
    tenantId: text("tenant_id").notNull(),
    personaId: text("persona_id")
      .notNull()
      .references(() => personaProfiles.id, { onDelete: "cascade" }),
    targetId: text("target_id").notNull(),
    targetKind: text("target_kind")
      .notNull()
      .$type<PersonaEmbeddingTargetKind>(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    uniqueIndex("idx_persona_memory_embeddings_target").on(
      table.targetKind,
      table.targetId
    ),
    index("idx_persona_memory_embeddings_persona").on(
      table.tenantId,
      table.personaId
    ),
  ]
);

export const personaAuditLogs = pgTable(
  "persona_audit_logs",
  {
    action: text("action").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    id: text("id").primaryKey().$defaultFn(ulid),
    metadata: jsonb("metadata").$type<Record<string, unknown>>().notNull(),
    tenantId: text("tenant_id").notNull(),
    personaId: text("persona_id").references(() => personaProfiles.id, {
      onDelete: "set null",
    }),
    userId: text("user_id"),
  },
  (table) => [
    index("idx_persona_audit_logs_persona").on(
      table.tenantId,
      table.personaId
    ),
    index("idx_persona_audit_logs_created").on(table.createdAt),
  ]
);

export type PersonaProfile = typeof personaProfiles.$inferSelect;
export type NewPersonaProfile = typeof personaProfiles.$inferInsert;
export type PersonaAlias = typeof personaAliases.$inferSelect;
export type NewPersonaAlias = typeof personaAliases.$inferInsert;
export type PersonaSourceDocument = typeof personaSourceDocuments.$inferSelect;
export type NewPersonaSourceDocument =
  typeof personaSourceDocuments.$inferInsert;
export type PersonaSourceChunk = typeof personaSourceChunks.$inferSelect;
export type NewPersonaSourceChunk = typeof personaSourceChunks.$inferInsert;
export type PersonaFact = typeof personaFacts.$inferSelect;
export type NewPersonaFact = typeof personaFacts.$inferInsert;
export type PersonaEpisodeMemory = typeof personaEpisodeMemories.$inferSelect;
export type NewPersonaEpisodeMemory =
  typeof personaEpisodeMemories.$inferInsert;
export type PersonaEmotionalSalience =
  typeof personaEmotionalSalience.$inferSelect;
export type NewPersonaEmotionalSalience =
  typeof personaEmotionalSalience.$inferInsert;
export type PersonaSemanticBelief = typeof personaSemanticBeliefs.$inferSelect;
export type NewPersonaSemanticBelief =
  typeof personaSemanticBeliefs.$inferInsert;
export type PersonaHabitPattern = typeof personaHabitPatterns.$inferSelect;
export type NewPersonaHabitPattern = typeof personaHabitPatterns.$inferInsert;
export type PersonaStyleProfile = typeof personaStyleProfiles.$inferSelect;
export type NewPersonaStyleProfile = typeof personaStyleProfiles.$inferInsert;
export type PersonaWorkspaceState = typeof personaWorkspaceStates.$inferSelect;
export type NewPersonaWorkspaceState =
  typeof personaWorkspaceStates.$inferInsert;
export type PersonaInteractionMemory =
  typeof personaInteractionMemories.$inferSelect;
export type NewPersonaInteractionMemory =
  typeof personaInteractionMemories.$inferInsert;
export type PersonaExternalMemoryRef =
  typeof personaExternalMemoryRefs.$inferSelect;
export type NewPersonaExternalMemoryRef =
  typeof personaExternalMemoryRefs.$inferInsert;
export type PersonaLifecycleEffect =
  typeof personaLifecycleEffects.$inferSelect;
export type NewPersonaLifecycleEffect =
  typeof personaLifecycleEffects.$inferInsert;
export type PersonaMemoryEmbedding =
  typeof personaMemoryEmbeddings.$inferSelect;
export type NewPersonaMemoryEmbedding =
  typeof personaMemoryEmbeddings.$inferInsert;
export type PersonaMoodState = typeof personaMoodStates.$inferSelect;
export type NewPersonaMoodState = typeof personaMoodStates.$inferInsert;
export type PersonaAuditLog = typeof personaAuditLogs.$inferSelect;
export type NewPersonaAuditLog = typeof personaAuditLogs.$inferInsert;
