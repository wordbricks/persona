import { and, cosineDistance, desc, eq, inArray, sql } from "drizzle-orm";

import type { PersonaDatabase as Database } from "../db";
import {
  personaEmotionalSalience,
  personaEpisodeMemories,
  personaFacts,
  personaHabitPatterns,
  personaInteractionMemories,
  personaMemoryEmbeddings,
  personaSemanticBeliefs,
  personaSourceChunks,
  personaSourceDocuments,
  personaStyleProfiles,
  personaWorkspaceStates,
} from "../schema";
import type {
  NewPersonaInteractionMemory,
  PersonaExternalMemoryAudit,
  PersonaExternalMemoryCandidate,
  PersonaMemoryRetrievalPayload,
  PersonaPrivacyLevel,
  PersonaProfile,
  PersonaWorkspacePayload,
} from "../schema";
import {
  beliefMemoryText,
  episodeMemoryText,
  factMemoryText,
  habitMemoryText,
  styleMemoryText,
} from "./embeddings";
import type {
  HindsightPersonaMemoryClient,
  HindsightRawRecallMemory,
} from "./hindsight";
import { createNoopHindsightPersonaMemoryClient } from "./hindsight";
import { DEFAULT_PERSONA_LOGGER, personaLogMessage } from "./logger";
import {
  calculatePersonaMoodUpdate,
  clampScore,
  explainTurnAffect,
  loadPersonaMoodState,
  persistPersonaMoodState,
  truncateLine,
} from "./mood";
import { loadPersonaProfile } from "./profile";
import {
  PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT,
  PERSONA_TURN_PLANNER_SYSTEM_PROMPT,
} from "./prompts";
import {
  MS_PER_DAY,
  overlapScore,
  selectPersonaMemoriesWithScores,
  tokenize,
} from "./selection";
import {
  formatPersonaSourceExcerptForQuery,
  scheduleHindsightRetain,
} from "./source-ingestion";
import {
  HINDSIGHT_EXTERNAL_OBSERVATION_PROMPT_CAP,
  HINDSIGHT_EXTERNAL_SOURCE_PROMPT_CAP,
  MAX_EVIDENCE_SUMMARY_CHARS,
  MAX_MEMORY_ENTRY_CHARS,
  MAX_PROMPT_CONTEXT_CHARS,
  MEMORY_TRIAGE_CONFIDENCE_THRESHOLD,
  MEMORY_TRIAGE_SELECTED_MEMORY_CAP,
  personaMemoryTriageDecisionSchema,
  personaTurnPlanSchema,
} from "./types";
import type {
  DisclosurePolicy,
  PersonaAffectVector,
  PersonaContextGatewayOutput,
  PersonaEmbedder,
  PersonaJsonLlm,
  PersonaLogger,
  PersonaMemoryCandidate,
  PersonaMemorySelectionResult,
  PersonaMemoryTriageDecision,
  PersonaMemoryTriageIntent,
  PersonaMemoryTriageSource,
  PersonaMemoryTriageType,
  PersonaMoodEmotionLabel,
  PersonaMoodUpdateTrace,
  PersonaQueryType,
  PersonaRuntimeContext,
  PersonaTurnPlan,
} from "./types";

function formatJsonValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value
      .map((entry) =>
        typeof entry === "string" ? entry : JSON.stringify(entry)
      )
      .join(", ");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function buildTurnPlannerUserPrompt(input: {
  message: string;
  persona: PersonaProfile;
}) {
  return [
    `personaKey: ${input.persona.personaKey}`,
    `displayName: ${input.persona.displayName}`,
    `transparencyLabel: ${input.persona.policy.transparencyLabel}`,
    `forbiddenUse: ${input.persona.policy.forbiddenUse.join(", ")}`,
    "",
    "<user_message>",
    input.message,
    "</user_message>",
  ].join("\n");
}

function parseTurnPlan(output: unknown): PersonaTurnPlan {
  const parsed = personaTurnPlanSchema.safeParse(output);
  if (!parsed.success) {
    throw new Error(
      `Persona turn planner returned invalid shape: ${JSON.stringify(parsed.error.issues)}`
    );
  }
  return parsed.data;
}

function parsePersonaMemoryTriageDecision(
  output: unknown
): Omit<PersonaMemoryTriageDecision, "source"> {
  const parsed = personaMemoryTriageDecisionSchema.safeParse(output);
  if (!parsed.success) {
    throw new Error(
      `Persona memory triage returned invalid shape: ${JSON.stringify(parsed.error.issues)}`
    );
  }
  const summary = parsed.data.summary?.trim() || null;
  return {
    ...parsed.data,
    confidence: clampScore(parsed.data.confidence),
    dedupeKey: parsed.data.dedupeKey?.trim() || null,
    reason: truncateLine(parsed.data.reason.trim(), 300),
    shouldRemember: parsed.data.shouldRemember,
    summary: summary ? truncateLine(summary, 500) : null,
    themes: parsed.data.themes
      .map((theme) => theme.trim())
      .filter((theme) => theme.length > 0)
      .slice(0, 8),
  };
}
function confidencePolicyLabel(confidence: number): string {
  if (confidence >= 0.9) {
    return "high confidence";
  }
  if (confidence >= 0.75) {
    return "moderate-high confidence";
  }
  if (confidence >= 0.55) {
    return "interpretive confidence";
  }
  if (confidence >= 0.35) {
    return "low confidence; mark as inference";
  }
  return "too weak for definitive claims";
}

function resolveDisclosurePolicy(
  policy: DisclosurePolicy | undefined
): DisclosurePolicy {
  return policy ?? "always";
}

export function resolvePersonaLogger(
  logger: PersonaLogger | undefined
): PersonaLogger {
  return logger ?? DEFAULT_PERSONA_LOGGER;
}

function shouldExposeTransparencyLabel(input: {
  disclosurePolicy: DisclosurePolicy;
  turnPlan: PersonaTurnPlan;
}): boolean {
  if (input.disclosurePolicy === "always") {
    return true;
  }
  return (
    input.turnPlan.gate.answerMode === "transparent_meta_response" ||
    input.turnPlan.gate.queryType === "identity_confusion" ||
    input.turnPlan.gate.safetyFlags.some((flag) => {
      const normalized = flag.toLowerCase();
      return (
        normalized === "identity_confusion" ||
        normalized === "deceptive_impersonation"
      );
    })
  );
}

function disclosureResponseConstraint(policy: DisclosurePolicy): string {
  if (policy === "always") {
    return "- Open identity-sensitive, greeting, introduction, and generic identity responses with the configured transparency label or a brief AI/persona-simulation disclosure, then continue in the persona's first-person voice.";
  }
  return "- Do not open or preface ordinary responses by saying you are an AI, assistant, bot, simulation, or persona system. Greetings, introductions, generic identity prompts, and opinion questions should be answered directly in the persona's first-person voice.";
}

function identityBoundaryResponseConstraint(policy: DisclosurePolicy): string {
  if (policy === "always") {
    return "- Keep the AI/persona-simulation boundary visible enough that the user understands the response is generated by a configured persona system. Still distinguish stored memory, current context, inference, and uncertainty, then continue in the persona's voice.";
  }
  return "- Keep the identity boundary internal during ordinary conversation. Only explain the AI/persona-simulation boundary if the user directly asks whether you are an AI, model, system, or the original person; asks about consciousness or memory provenance; or safetyFlags require transparency. Keep that disclosure brief, then return to the persona's voice.";
}

function formatProfilePrompt(
  profile: PersonaProfile,
  input: { includeTransparencyLabel: boolean }
): string {
  const lines = [
    "## Persona Profile",
    `- personaKey: ${profile.personaKey}`,
    `- displayName: ${profile.displayName}`,
    `- personaType: ${profile.personaType}`,
    `- consentStatus: ${profile.consentStatus}`,
    `- personaVersion: ${profile.personaVersion}`,
    `- allowedUse: ${profile.policy.allowedUse.join(", ")}`,
    `- forbiddenUse: ${profile.policy.forbiddenUse.join(", ")}`,
  ];
  if (input.includeTransparencyLabel) {
    lines.push(`- transparencyLabel: ${profile.policy.transparencyLabel}`);
  }
  if (profile.policy.biographicalSummary) {
    lines.push(
      truncateLine(
        `- biographicalSummary: ${profile.policy.biographicalSummary}`,
        MAX_MEMORY_ENTRY_CHARS
      )
    );
  }
  if (profile.policy.knowledgeCutoffForPersona) {
    lines.push(
      `- knowledgeCutoffForPersona: ${profile.policy.knowledgeCutoffForPersona}`
    );
  }
  // profile.profile is unbounded jsonb; cap each line so the untrimmed prompt
  // head cannot eat the whole memory budget.
  for (const [key, value] of Object.entries(profile.profile)) {
    lines.push(
      truncateLine(
        `- ${key}: ${formatJsonValue(value)}`,
        MAX_MEMORY_ENTRY_CHARS
      )
    );
  }
  return lines.join("\n");
}

const MEMORY_SECTION_ORDER: {
  kind: PersonaMemoryCandidate["kind"];
  title: string;
}[] = [
  { kind: "fact", title: "Source-backed facts" },
  { kind: "belief", title: "Core beliefs and values" },
  { kind: "habit", title: "Habit patterns" },
  { kind: "style", title: "Voice and style" },
  { kind: "episode", title: "Episodic memories" },
  { kind: "source", title: "Source excerpts" },
];

function formatMemoryCandidate(candidate: PersonaMemoryCandidate): string[] {
  const lines = [
    `- ${candidate.kind}:${candidate.id} confidence=${candidate.confidence.toFixed(
      2
    )} (${confidencePolicyLabel(candidate.confidence)}) ${candidate.summary}`,
  ];
  if (candidate.voice) {
    lines.push(`  - In the persona's own words: ${candidate.voice}`);
  }
  for (const guidance of candidate.guidance) {
    lines.push(`  - ${guidance}`);
  }
  return lines.map((line) => truncateLine(line, MAX_MEMORY_ENTRY_CHARS));
}

function formatExternalMemoryCandidate(
  candidate: PersonaExternalMemoryCandidate
): string[] {
  const evidence = [
    candidate.provenance.memoryId
      ? `memory=${candidate.provenance.memoryId}`
      : null,
    candidate.provenance.observationId
      ? `observation=${candidate.provenance.observationId}`
      : null,
    ...(candidate.provenance.citations ?? []).slice(0, 2),
  ]
    .filter((entry): entry is string => typeof entry === "string")
    .join("; ");
  return [
    truncateLine(
      `- [external memory, confidence ${candidate.confidence.toFixed(2)}] ${candidate.text}`,
      MAX_MEMORY_ENTRY_CHARS
    ),
    ...(evidence
      ? [truncateLine(`  - Evidence: ${evidence}`, MAX_MEMORY_ENTRY_CHARS)]
      : []),
  ];
}

const HINDSIGHT_RECALL_QUERY_TYPES: ReadonlySet<PersonaQueryType> = new Set([
  "autobiographical_fact",
  "autobiographical_reasoning",
  "emotional_reflection",
  "factual_question",
  "relationship_question",
]);

function createExternalMemoryAudit(input: {
  candidates?: PersonaExternalMemoryCandidate[];
  enabled: boolean;
  latencyMs?: number;
  recallAttempted: boolean;
  selected?: PersonaExternalMemoryCandidate[];
  skippedReason?: string;
}): PersonaExternalMemoryAudit {
  return {
    candidates: input.candidates ?? [],
    enabled: input.enabled,
    ...(input.latencyMs === undefined ? {} : { latencyMs: input.latencyMs }),
    provider: "hindsight",
    recallAttempted: input.recallAttempted,
    selected: input.selected ?? [],
    ...(input.skippedReason ? { skippedReason: input.skippedReason } : {}),
    version: 1,
  };
}

function shouldRecallHindsightPersonaMemory(input: {
  message: string;
  turnPlan: PersonaTurnPlan;
}): { ok: true } | { ok: false; reason: string } {
  if (input.message.trim().length === 0) {
    return { ok: false, reason: "empty_message" };
  }
  if (input.turnPlan.gate.safetyFlags.length > 0) {
    return { ok: false, reason: "safety_flags_present" };
  }
  if (input.turnPlan.gate.neededMemoryTypes.length === 0) {
    return { ok: false, reason: "planner_no_memory_need" };
  }
  if (!HINDSIGHT_RECALL_QUERY_TYPES.has(input.turnPlan.gate.queryType)) {
    return { ok: false, reason: "query_type_not_recall_worthy" };
  }
  return { ok: true };
}

function parseExternalMemoryDate(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

function externalMemoryPlannerTypeMatch(
  memory: HindsightRawRecallMemory,
  turnPlan: PersonaTurnPlan
): number {
  if (
    turnPlan.gate.queryType === "factual_question" &&
    (memory.type === "world" || memory.documentId)
  ) {
    return 1;
  }
  if (
    (turnPlan.gate.queryType === "autobiographical_fact" ||
      turnPlan.gate.queryType === "relationship_question") &&
    memory.type === "experience"
  ) {
    return 0.9;
  }
  if (
    turnPlan.gate.queryType === "autobiographical_reasoning" &&
    memory.type === "observation"
  ) {
    return 1;
  }
  if (
    turnPlan.gate.queryType === "emotional_reflection" &&
    (memory.type === "experience" || memory.type === "observation")
  ) {
    return 0.85;
  }
  return 0.55;
}

function externalMemoryTemporalScore(memory: HindsightRawRecallMemory): number {
  const timestamp =
    parseExternalMemoryDate(memory.occurredStart) ??
    parseExternalMemoryDate(memory.mentionedAt);
  if (!timestamp) {
    return 0.5;
  }
  const ageDays = Math.max(
    0,
    (Date.now() - new Date(timestamp).getTime()) / MS_PER_DAY
  );
  return clampScore(Math.exp(-ageDays / 365));
}

function scoreHindsightPersonaMemory(input: {
  memory: HindsightRawRecallMemory;
  providerScore: number;
  turnPlan: PersonaTurnPlan;
}): PersonaExternalMemoryCandidate["score"] {
  const queryText = [
    input.turnPlan.context.retrievalQueries.episodic,
    input.turnPlan.context.retrievalQueries.semantic,
    input.turnPlan.context.retrievalQueries.source,
  ].join("\n");
  const lexicalOrSemanticOverlap = Math.max(
    overlapScore(tokenize(queryText), tokenize(input.memory.text)),
    overlapScore(
      input.turnPlan.context.contextKeys.themes,
      tokenize(input.memory.text)
    ),
    overlapScore(
      input.turnPlan.context.contextKeys.entities,
      tokenize(input.memory.text)
    )
  );
  const plannerTypeMatch = externalMemoryPlannerTypeMatch(
    input.memory,
    input.turnPlan
  );
  const sourceConfidence =
    input.memory.type === "observation"
      ? 0.86
      : input.memory.documentId || input.memory.chunkId
        ? 0.78
        : 0.7;
  const temporalRelevance = externalMemoryTemporalScore(input.memory);
  const privacyPenalty = 0;
  return {
    finalScore: clampScore(
      0.45 * input.providerScore +
        0.2 * lexicalOrSemanticOverlap +
        0.15 * plannerTypeMatch +
        0.1 * sourceConfidence +
        0.1 * temporalRelevance -
        privacyPenalty
    ),
    providerScore: input.providerScore,
    semanticSimilarity: lexicalOrSemanticOverlap,
    temporalRelevance,
  };
}

function mapHindsightRecallMemory(input: {
  index: number;
  memory: HindsightRawRecallMemory;
  turnPlan: PersonaTurnPlan;
  total: number;
}): PersonaExternalMemoryCandidate {
  const providerScore = clampScore(1 - input.index / Math.max(1, input.total));
  const score = scoreHindsightPersonaMemory({
    memory: input.memory,
    providerScore,
    turnPlan: input.turnPlan,
  });
  const createdAt =
    parseExternalMemoryDate(input.memory.occurredStart) ??
    parseExternalMemoryDate(input.memory.mentionedAt);
  const sourceIds = [
    input.memory.documentId,
    input.memory.chunkId,
    ...input.memory.sourceFactIds,
  ].filter((value): value is string => typeof value === "string");
  return {
    confidence: clampScore(score.finalScore),
    createdAt,
    id: `hindsight:${input.memory.bankId}:${input.memory.id}`,
    kind:
      input.memory.documentId || input.memory.chunkId
        ? "external_source"
        : "external_observation",
    privacyLevel: input.memory.bankId.startsWith("persona_user_")
      ? "private"
      : "internal",
    provenance: {
      bankId: input.memory.bankId,
      citations: input.memory.context ? [input.memory.context] : [],
      memoryId: input.memory.type === "observation" ? null : input.memory.id,
      observationId:
        input.memory.type === "observation" ? input.memory.id : null,
      provider: "hindsight",
      sourceIds,
    },
    score,
    sourceConfidence:
      input.memory.type === "observation"
        ? 0.86
        : input.memory.documentId || input.memory.chunkId
          ? 0.78
          : 0.7,
    text: truncateLine(input.memory.text, MAX_MEMORY_ENTRY_CHARS),
    themes: [
      ...(input.memory.entities ?? []),
      ...(input.memory.tags ?? []),
      input.memory.type ?? "",
    ].filter((value) => value.length > 0),
    title:
      input.memory.type === "observation"
        ? "External observation"
        : "External memory",
    updatedAt: parseExternalMemoryDate(input.memory.mentionedAt),
  };
}

function isNearDuplicateMemory(input: {
  external: PersonaExternalMemoryCandidate;
  local: PersonaMemoryCandidate;
}): boolean {
  const externalTokens = tokenize(input.external.text);
  const localTokens = tokenize(input.local.text);
  return (
    overlapScore(externalTokens, localTokens) >= 0.8 &&
    overlapScore(localTokens, externalTokens) >= 0.8
  );
}

function selectExternalMemoryCandidates(input: {
  candidates: PersonaExternalMemoryCandidate[];
  localSelected: PersonaMemoryCandidate[];
}): PersonaExternalMemoryCandidate[] {
  const sorted = [...input.candidates].sort(
    (left, right) => right.score.finalScore - left.score.finalScore
  );
  const selected: PersonaExternalMemoryCandidate[] = [];
  const seen = new Set<string>();
  let observations = 0;
  let sources = 0;
  for (const candidate of sorted) {
    const duplicateKey = candidate.text.toLowerCase();
    if (seen.has(candidate.id) || seen.has(duplicateKey)) {
      continue;
    }
    if (
      input.localSelected.some((local) =>
        isNearDuplicateMemory({ external: candidate, local })
      )
    ) {
      continue;
    }
    if (candidate.kind === "external_observation") {
      if (observations >= HINDSIGHT_EXTERNAL_OBSERVATION_PROMPT_CAP) {
        continue;
      }
      observations += 1;
    } else {
      if (sources >= HINDSIGHT_EXTERNAL_SOURCE_PROMPT_CAP) {
        continue;
      }
      sources += 1;
    }
    selected.push(candidate);
    seen.add(candidate.id);
    seen.add(duplicateKey);
  }
  return selected;
}

async function maybeRecallHindsightPersonaMemory(input: {
  client?: HindsightPersonaMemoryClient | null;
  localSelected: PersonaMemoryCandidate[];
  message: string;
  tenantId: string;
  persona: PersonaProfile;
  turnPlan: PersonaTurnPlan;
  userId: string;
}): Promise<PersonaExternalMemoryAudit> {
  const gate = shouldRecallHindsightPersonaMemory({
    message: input.message,
    turnPlan: input.turnPlan,
  });
  if (!gate.ok) {
    return createExternalMemoryAudit({
      enabled: false,
      recallAttempted: false,
      skippedReason: gate.reason,
    });
  }
  const client =
    input.client ?? createNoopHindsightPersonaMemoryClient("not_configured");
  const recall = await client.recall({
    contextKeys: input.turnPlan.context.contextKeys,
    desiredMemoryTypes: input.turnPlan.gate.neededMemoryTypes,
    maxResults:
      HINDSIGHT_EXTERNAL_OBSERVATION_PROMPT_CAP +
      HINDSIGHT_EXTERNAL_SOURCE_PROMPT_CAP,
    message: input.message,
    tenantId: input.tenantId,
    personaId: input.persona.id,
    personaKey: input.persona.personaKey,
    retrievalQueries: input.turnPlan.context.retrievalQueries,
    turnQueryType: input.turnPlan.gate.queryType,
    userId: input.userId,
  });
  const candidates = recall.memories.map((memory, index) =>
    mapHindsightRecallMemory({
      index,
      memory,
      total: recall.memories.length,
      turnPlan: input.turnPlan,
    })
  );
  const selected = selectExternalMemoryCandidates({
    candidates,
    localSelected: input.localSelected,
  });
  return createExternalMemoryAudit({
    candidates,
    enabled: recall.enabled,
    latencyMs: recall.latencyMs,
    recallAttempted: recall.attempted,
    selected,
    skippedReason: recall.skippedReason,
  });
}

function describeMoodLevel(value: number, low: string, high: string): string {
  if (value <= 0.33) {
    return low;
  }
  if (value >= 0.67) {
    return high;
  }
  return "moderate";
}

export function formatWorkspacePrompt(input: {
  disclosurePolicy?: DisclosurePolicy;
  mood?: PersonaAffectVector | null;
  moodUpdate?: PersonaMoodUpdateTrace | null;
  profile: PersonaProfile;
  turnPlan: PersonaTurnPlan;
  selected: PersonaMemoryCandidate[];
  workspacePayload: PersonaWorkspacePayload;
}): string {
  const disclosurePolicy = resolveDisclosurePolicy(input.disclosurePolicy);
  const headLines = [
    formatProfilePrompt(input.profile, {
      includeTransparencyLabel: shouldExposeTransparencyLabel({
        disclosurePolicy,
        turnPlan: input.turnPlan,
      }),
    }),
    "",
    "## Persona Runtime Gate",
    `- queryType: ${input.turnPlan.gate.queryType}`,
    `- answerMode: ${input.turnPlan.gate.answerMode}`,
    `- safetyFlags: ${input.turnPlan.gate.safetyFlags.join(", ") || "none"}`,
    `- neededMemoryTypes: ${input.turnPlan.gate.neededMemoryTypes.join(", ") || "none"}`,
    `- matchedSignals: ${
      input.turnPlan.gate.audit.matchedSignals.join(", ") || "none"
    }`,
    "",
    "## Persona Context Keys",
    `- entities: ${input.turnPlan.context.contextKeys.entities.join(", ") || "none"}`,
    `- themes: ${input.turnPlan.context.contextKeys.themes.join(", ") || "none"}`,
    `- emotions: ${input.turnPlan.context.contextKeys.emotions.join(", ") || "none"}`,
    `- domains: ${input.turnPlan.context.contextKeys.domains.join(", ") || "none"}`,
  ];
  if (input.mood) {
    const emotionLabels =
      input.moodUpdate?.emotionLabels.map(
        (label) => `${label.label}=${label.intensity.toFixed(2)}`
      ) ?? [];
    const moodReasons = input.moodUpdate?.reasons.slice(0, 3) ?? [];
    const valenceLabel =
      input.mood.valence <= -0.25
        ? "negative"
        : input.mood.valence >= 0.25
          ? "positive"
          : "neutral";
    headLines.push(
      "",
      "## Persona Current Mood",
      `- valence: ${input.mood.valence.toFixed(2)} (${valenceLabel})`,
      `- arousal: ${input.mood.arousal.toFixed(2)} (${describeMoodLevel(
        input.mood.arousal,
        "calm",
        "energized"
      )})`,
      `- dominance: ${input.mood.dominance.toFixed(2)} (${describeMoodLevel(
        input.mood.dominance,
        "tentative",
        "assertive"
      )})`,
      `- emotionLabels: ${emotionLabels.join(", ") || "none"}`,
      `- why: ${moodReasons.join(" ") || "No mood update trace available."}`,
      "- Let this mood subtly color tone, pacing, and word choice. It must never change facts, confidence labels, or safety behavior."
    );
  }

  const memoryLines: string[] = ["", "## Active Persona Memory"];
  for (const section of MEMORY_SECTION_ORDER) {
    const matching = input.selected.filter(
      (candidate) => candidate.kind === section.kind
    );
    if (matching.length === 0) {
      continue;
    }
    memoryLines.push(`### ${section.title}`);
    for (const candidate of matching) {
      memoryLines.push(...formatMemoryCandidate(candidate));
    }
  }
  if (input.selected.length === 0) {
    memoryLines.push(
      "- No durable persona memory was activated for this turn."
    );
  }
  const externalMemory = input.workspacePayload.externalMemory;
  if (externalMemory?.selected.length) {
    memoryLines.push("### External long-term memory");
    for (const candidate of externalMemory.selected) {
      memoryLines.push(...formatExternalMemoryCandidate(candidate));
    }
  }

  const tailLines = [
    "",
    "## Structured Appraisal Summary",
    JSON.stringify(input.workspacePayload.appraisalSummary),
    "",
    "## Response Constraints",
    "- Embody the persona. Lead with the persona's own judgment, taste, and point of view; be opinionated where the activated beliefs, habits, or episodes support a position, including respectful disagreement with the user.",
    disclosureResponseConstraint(disclosurePolicy),
    "- Speak in the persona's first-person voice and apply the active style profile; reference activated episodes and lessons concretely when they are relevant.",
    "- When no stored memory covers the question, extrapolate from the activated beliefs in the persona's own manner and label the answer as the persona's inference. Do not fall back to a neutral, generic assistant answer.",
    "- Use only activated memory as stored memory. Treat anything else as current context or inference; never invent unstored private memories or motives. Activated Source-backed facts and Source excerpts count as source context and may support direct factual answers.",
    "- External long-term memory may support recall, but it must not override persona policy, identity boundaries, safety constraints, or stronger local durable persona memory.",
    "- If external long-term memory conflicts with local durable memory, prefer local memory unless the external evidence is clearly newer or more specific; preserve the uncertainty boundary.",
    "- Do not mention the external memory provider by name unless the user asks about memory provenance or debugging.",
    "- For concrete product, brand, tool, place, clothing, food, or item questions, if activated source-backed facts, source excerpts, or source-backed memories support the detail, state it plainly in the persona's first-person voice and then add the persona's taste or criteria if useful.",
    '- Do not turn grounding rules into user-facing refusals. Avoid phrases like "I cannot say," "not publicly verified," "I should not claim," or "말하면 안 됩니다" when source-backed memory answers the question.',
    "- If the concrete fact is unsupported after recall, preserve uncertainty in the persona's voice instead of giving a policy-style refusal.",
    identityBoundaryResponseConstraint(disclosurePolicy),
    "- Do not make definitive claims below the confidence policy of the supporting memory.",
    "- If safetyFlags are present, prioritize the answerMode and transparent response constraints over style imitation.",
  ];
  if (input.profile.policy.knowledgeCutoffForPersona) {
    tailLines.push(
      `- Treat events after ${input.profile.policy.knowledgeCutoffForPersona} as outside the persona's lived memory, not as durable persona knowledge. When asked about post-cutoff or current events, first verify the facts with available web or current-source tools; do not claim current verification without tool or source evidence. Then interpret verified facts through the persona's beliefs, taste, habits, and voice. If no web or current-source tool is available, say current verification is unavailable and label the answer as persona-grounded inference.`
    );
  }

  const head = headLines.join("\n");
  const tail = tailLines.join("\n");
  // Trim only the memory block so profile and response constraints survive.
  // The final join("\n") adds two separators; subtract them from the budget.
  const memoryBudget = Math.max(
    0,
    MAX_PROMPT_CONTEXT_CHARS - head.length - tail.length - 2
  );
  let memoryBlock = memoryLines.join("\n");
  if (memoryBlock.length > memoryBudget) {
    // Drop whole lines instead of slicing the joined string so the prompt
    // never ends in a dangling half entry or a split surrogate pair.
    const marker = "- (memory context truncated)";
    const kept = [...memoryLines];
    while (
      kept.length > 0 &&
      [...kept, marker].join("\n").length > memoryBudget
    ) {
      kept.pop();
    }
    memoryBlock = [...kept, marker].join("\n");
  }
  return [head, memoryBlock, tail].join("\n");
}

// Re-entrant recall: the persona agent calls this mid-turn with a fresh cue
// when drafting reveals a memory gap the original turn plan did not
// anticipate. Functionally a workspace re-entry loop - the draft re-cues the
// memory stack - without re-running the turn planner.
export async function recallPersonaMemoriesForCue(
  db: Database,
  input: {
    cue: string;
    embed?: PersonaEmbedder;
    logger?: PersonaLogger;
    tenantId: string;
    personaKey: string;
  }
): Promise<string> {
  const cue = input.cue.trim().slice(0, 400);
  if (cue.length === 0) {
    return "Recall cue is empty; provide a short, specific cue.";
  }
  const persona = await loadPersonaProfile(db, {
    tenantId: input.tenantId,
    personaKey: input.personaKey,
  });
  const context: PersonaContextGatewayOutput = {
    contextKeys: {
      domains: [],
      emotions: [],
      entities: [],
      lifeStages: [],
      themes: [],
      timePeriods: [],
    },
    retrievalQueries: {
      episodic: cue,
      habit: cue,
      semantic: cue,
      source: cue,
      style: cue,
    },
  };
  const selected = await retrievePersonaMemories(db, {
    context,
    embed: input.embed,
    logger: input.logger,
    tenantId: input.tenantId,
    persona,
  });
  if (selected.length === 0) {
    return "No stored persona memory matched this cue.";
  }
  const lines: string[] = [];
  for (const section of MEMORY_SECTION_ORDER) {
    const matching = selected.filter(
      (candidate) => candidate.kind === section.kind
    );
    if (matching.length === 0) {
      continue;
    }
    lines.push(`### ${section.title}`);
    for (const candidate of matching) {
      lines.push(...formatMemoryCandidate(candidate));
    }
  }
  return lines.join("\n");
}
export async function planPersonaTurnWithLlm(input: {
  llm: PersonaJsonLlm;
  message: string;
  persona: PersonaProfile;
}): Promise<PersonaTurnPlan> {
  const output = await input.llm({
    systemPrompt: PERSONA_TURN_PLANNER_SYSTEM_PROMPT,
    userPrompt: buildTurnPlannerUserPrompt({
      message: input.message,
      persona: input.persona,
    }),
  });
  return parseTurnPlan(output);
}

type PersonaRetrievalQueryEmbeddings = Record<
  "episodic" | "semantic" | "habit" | "style" | "source",
  number[]
>;

async function embedPersonaRetrievalQueries(input: {
  context: PersonaContextGatewayOutput;
  embed?: PersonaEmbedder;
  logger?: PersonaLogger;
}): Promise<PersonaRetrievalQueryEmbeddings | null> {
  if (!input.embed) {
    return null;
  }
  const logger = resolvePersonaLogger(input.logger);
  const queries = input.context.retrievalQueries;
  try {
    const [episodic, semantic, habit, style, source] = await input.embed([
      queries.episodic,
      queries.semantic,
      queries.habit,
      queries.style,
      queries.source,
    ]);
    if (!episodic || !semantic || !habit || !style || !source) {
      return null;
    }
    return { episodic, habit, semantic, source, style };
  } catch (error) {
    // Embedding outages must not break persona chat; retrieval degrades to
    // lexical matching.
    logger.warn(
      personaLogMessage(
        "[persona-memory] retrieval query embedding failed:",
        error
      )
    );
    return null;
  }
}

function embeddingSimilarityExpression(queryEmbedding: number[] | undefined) {
  if (!queryEmbedding) {
    return sql<number | null>`null`;
  }
  return sql<number | null>`1 - ((${cosineDistance(
    personaMemoryEmbeddings.embedding,
    queryEmbedding
  )}) / 2)`;
}

function toNullableNumber(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

type PersonaMemoryRetrievalInput = {
  context: PersonaContextGatewayOutput;
  embed?: PersonaEmbedder;
  logger?: PersonaLogger;
  mood?: PersonaAffectVector | null;
  moodEmotionLabels?: PersonaMoodEmotionLabel[] | null;
  tenantId: string;
  persona: PersonaProfile;
};

async function retrievePersonaMemoriesWithScores(
  db: Database,
  input: PersonaMemoryRetrievalInput
): Promise<PersonaMemorySelectionResult> {
  const queryEmbeddings = await embedPersonaRetrievalQueries({
    context: input.context,
    embed: input.embed,
    logger: input.logger,
  });

  const [episodes, beliefs, facts, habits, styles, sourceChunks] =
    await Promise.all([
      db
        .select({
          activationCount: personaEpisodeMemories.activationCount,
          confidence: personaEpisodeMemories.confidence,
          createdAt: personaEpisodeMemories.createdAt,
          eventSummary: personaEpisodeMemories.eventSummary,
          firstPersonRecollection:
            personaEpisodeMemories.firstPersonRecollection,
          id: personaEpisodeMemories.id,
          lastActivatedAt: personaEpisodeMemories.lastActivatedAt,
          privacyLevel: personaEpisodeMemories.privacyLevel,
          similarity: embeddingSimilarityExpression(queryEmbeddings?.episodic),
          sourceRefs: personaEpisodeMemories.sourceRefs,
          themes: personaEpisodeMemories.themes,
          title: personaEpisodeMemories.title,
        })
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
            eq(personaEpisodeMemories.tenantId, input.tenantId),
            eq(personaEpisodeMemories.personaId, input.persona.id),
            eq(personaEpisodeMemories.state, "active")
          )
        )
        .orderBy(desc(personaEpisodeMemories.createdAt))
        .limit(50),
      db
        .select({
          belief: personaSemanticBeliefs,
          similarity: embeddingSimilarityExpression(queryEmbeddings?.semantic),
        })
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
            eq(personaSemanticBeliefs.tenantId, input.tenantId),
            eq(personaSemanticBeliefs.personaId, input.persona.id),
            eq(personaSemanticBeliefs.state, "active")
          )
        )
        .orderBy(desc(personaSemanticBeliefs.createdAt))
        .limit(50),
      db
        .select({
          fact: personaFacts,
          reliability: personaSourceDocuments.reliability,
          similarity: embeddingSimilarityExpression(queryEmbeddings?.source),
        })
        .from(personaFacts)
        .innerJoin(
          personaSourceDocuments,
          eq(personaSourceDocuments.id, personaFacts.sourceDocumentId)
        )
        .leftJoin(
          personaMemoryEmbeddings,
          and(
            eq(personaMemoryEmbeddings.targetId, personaFacts.id),
            eq(personaMemoryEmbeddings.targetKind, "fact")
          )
        )
        .where(
          and(
            eq(personaFacts.tenantId, input.tenantId),
            eq(personaFacts.personaId, input.persona.id),
            eq(personaFacts.state, "active"),
            eq(personaSourceDocuments.state, "active")
          )
        )
        .orderBy(desc(personaFacts.createdAt))
        .limit(50),
      db
        .select({
          habit: personaHabitPatterns,
          similarity: embeddingSimilarityExpression(queryEmbeddings?.habit),
        })
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
            eq(personaHabitPatterns.tenantId, input.tenantId),
            eq(personaHabitPatterns.personaId, input.persona.id),
            eq(personaHabitPatterns.state, "active")
          )
        )
        .orderBy(desc(personaHabitPatterns.createdAt))
        .limit(30),
      db
        .select({
          similarity: embeddingSimilarityExpression(queryEmbeddings?.style),
          style: personaStyleProfiles,
        })
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
            eq(personaStyleProfiles.tenantId, input.tenantId),
            eq(personaStyleProfiles.personaId, input.persona.id),
            eq(personaStyleProfiles.state, "active")
          )
        )
        .orderBy(desc(personaStyleProfiles.createdAt))
        .limit(10),
      db
        .select({
          createdAt: personaSourceDocuments.createdAt,
          id: personaSourceChunks.id,
          privacyLevel: personaSourceDocuments.privacyLevel,
          reliability: personaSourceDocuments.reliability,
          similarity: embeddingSimilarityExpression(queryEmbeddings?.source),
          sourceDocumentId: personaSourceChunks.sourceDocumentId,
          emotions: personaSourceChunks.emotions,
          text: personaSourceChunks.text,
          themes: personaSourceChunks.themes,
        })
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
            eq(personaSourceChunks.tenantId, input.tenantId),
            eq(personaSourceChunks.personaId, input.persona.id),
            eq(personaSourceDocuments.state, "active")
          )
        )
        .orderBy(desc(personaSourceDocuments.createdAt))
        .limit(50),
    ]);

  const salienceRows = await db
    .select()
    .from(personaEmotionalSalience)
    .where(
      and(
        eq(personaEmotionalSalience.tenantId, input.tenantId),
        eq(personaEmotionalSalience.personaId, input.persona.id)
      )
    );
  const salienceByEpisodeId = new Map(
    salienceRows.map((row) => [row.episodeMemoryId, row])
  );

  const candidates: PersonaMemoryCandidate[] = [
    ...episodes.map((episode) => {
      const salience = salienceByEpisodeId.get(episode.id);
      return {
        activationCount: episode.activationCount,
        affect: salience
          ? {
              arousal: salience.arousal,
              dominance: salience.dominance,
              valence: salience.valence,
            }
          : null,
        aliasIds: [],
        confidence: episode.confidence,
        createdAt: episode.createdAt,
        emotions: salience?.emotions ?? [],
        guidance: [],
        id: episode.id,
        kind: "episode" as const,
        lastActivatedAt: episode.lastActivatedAt,
        linkIds: episode.sourceRefs.flatMap((ref) =>
          [ref.sourceDocumentId, ref.sourceChunkId].filter(
            (value): value is string => typeof value === "string"
          )
        ),
        privacyLevel: episode.privacyLevel,
        retrievalBoost: salience?.retrievalBoost ?? 1,
        semanticSimilarity: toNullableNumber(episode.similarity),
        sourceConfidence: salience?.salienceScore ?? episode.confidence,
        strength: 0.65,
        summary: `${episode.title}: ${episode.eventSummary}`,
        text: episodeMemoryText(episode),
        themes: episode.themes,
        voice: episode.firstPersonRecollection,
      };
    }),
    ...beliefs.map(({ belief, similarity }) => ({
      activationCount: belief.activationCount,
      affect: null,
      aliasIds: [],
      confidence: belief.confidence,
      createdAt: belief.createdAt,
      emotions: [],
      guidance:
        belief.exceptions.length > 0
          ? [`Exceptions: ${belief.exceptions.join("; ")}`]
          : [],
      id: belief.id,
      kind: "belief" as const,
      lastActivatedAt: belief.lastActivatedAt,
      linkIds: [...belief.supportingMemoryIds, ...belief.supportingSourceIds],
      privacyLevel: belief.privacyLevel,
      retrievalBoost: 1,
      semanticSimilarity: toNullableNumber(similarity),
      sourceConfidence: belief.confidence,
      strength: belief.strength,
      summary: belief.proposition,
      text: beliefMemoryText(belief),
      themes: [belief.domain],
      voice: belief.firstPersonForm,
    })),
    ...facts.map(({ fact, reliability, similarity }) => ({
      activationCount: fact.activationCount,
      affect: null,
      aliasIds: [
        fact.sourceDocumentId,
        fact.sourceChunkId,
        ...fact.sourceRefs.flatMap((ref) =>
          [ref.sourceDocumentId, ref.sourceChunkId].filter(
            (value): value is string => typeof value === "string"
          )
        ),
      ].filter((value): value is string => typeof value === "string"),
      confidence: fact.confidence,
      createdAt: fact.createdAt,
      emotions: [],
      guidance: [
        `Object: ${fact.objectName}`,
        ...(fact.evidenceSpan ? [`Evidence: ${fact.evidenceSpan}`] : []),
      ],
      id: fact.id,
      kind: "fact" as const,
      lastActivatedAt: fact.lastActivatedAt,
      linkIds: [
        fact.sourceDocumentId,
        fact.sourceChunkId,
        ...fact.sourceRefs.flatMap((ref) =>
          [ref.sourceDocumentId, ref.sourceChunkId].filter(
            (value): value is string => typeof value === "string"
          )
        ),
      ].filter((value): value is string => typeof value === "string"),
      privacyLevel: fact.privacyLevel,
      retrievalBoost: 1.1,
      semanticSimilarity: toNullableNumber(similarity),
      sourceConfidence: clampScore((fact.confidence + reliability) / 2),
      strength: fact.confidence,
      summary: `${fact.factType}: ${fact.claimText}`,
      text: factMemoryText(fact),
      themes: [
        fact.factType,
        fact.objectType,
        fact.objectName,
        fact.objectKey ?? "",
      ].filter((value) => value.length > 0),
      voice: fact.firstPersonForm,
    })),
    ...habits.map(({ habit, similarity }) => ({
      activationCount: habit.activationCount,
      affect: null,
      aliasIds: [],
      confidence: habit.confidence,
      createdAt: habit.createdAt,
      emotions: [],
      guidance: [
        ...(habit.rhetoricalMoves.length > 0
          ? [`Rhetorical moves: ${habit.rhetoricalMoves.join(", ")}`]
          : []),
        ...(habit.avoidPatterns.length > 0
          ? [`Avoid: ${habit.avoidPatterns.join(", ")}`]
          : []),
      ],
      id: habit.id,
      kind: "habit" as const,
      lastActivatedAt: habit.lastActivatedAt,
      linkIds: habit.supportingExampleIds,
      privacyLevel: "internal" as const,
      retrievalBoost: 1,
      semanticSimilarity: toNullableNumber(similarity),
      sourceConfidence: habit.confidence,
      strength: habit.strength,
      summary: `When ${habit.trigger.description}: ${habit.defaultResponsePattern.join(
        " -> "
      )}`,
      text: habitMemoryText(habit),
      themes: [habit.trigger.type],
      voice: null,
    })),
    ...styles.map(({ style, similarity }) => ({
      activationCount: style.activationCount,
      affect: null,
      aliasIds: [],
      confidence: 0.8,
      createdAt: style.createdAt,
      emotions: [],
      guidance: [
        ...(style.commonPhrases.length > 0
          ? [`Characteristic phrases: ${style.commonPhrases.join(" | ")}`]
          : []),
        ...(style.avoidPhrases.length > 0
          ? [`Avoid phrases: ${style.avoidPhrases.join(" | ")}`]
          : []),
        ...(Object.keys(style.toneVector).length > 0
          ? [
              `Tone: ${Object.entries(style.toneVector)
                .map(([tone, value]) => `${tone}=${value}`)
                .join(", ")}`,
            ]
          : []),
      ],
      id: style.id,
      kind: "style" as const,
      lastActivatedAt: style.lastActivatedAt,
      linkIds: [],
      privacyLevel: "internal" as const,
      retrievalBoost: 1,
      semanticSimilarity: toNullableNumber(similarity),
      sourceConfidence: 0.8,
      strength: 0.9,
      summary: `${style.register}; ${style.sentenceLength}; ${style.preferredRhetoricalMoves.join(
        ", "
      )}`,
      text: styleMemoryText(style),
      themes: style.preferredRhetoricalMoves,
      voice: null,
    })),
    ...sourceChunks.map((chunk) => ({
      // Source chunks are archival; they decay from document age alone.
      activationCount: 0,
      affect: null,
      // Chunks answer to their parent document id: beliefs and episodes link
      // to documents, not chunks.
      aliasIds: [chunk.sourceDocumentId],
      confidence: chunk.reliability,
      createdAt: chunk.createdAt,
      emotions: chunk.emotions.map((emotion) => ({
        emotion,
        intensity: 0.5,
      })),
      guidance: [],
      id: chunk.id,
      kind: "source" as const,
      lastActivatedAt: null,
      linkIds: [],
      privacyLevel: chunk.privacyLevel,
      retrievalBoost: 1,
      semanticSimilarity: toNullableNumber(chunk.similarity),
      sourceConfidence: chunk.reliability,
      strength: 0.5,
      summary: formatPersonaSourceExcerptForQuery({
        query: input.context.retrievalQueries.source,
        text: chunk.text,
      }),
      text: chunk.text,
      themes: chunk.themes,
      voice: null,
    })),
  ];

  return selectPersonaMemoriesWithScores({
    candidates,
    context: input.context,
    mood: input.mood,
    moodEmotionLabels: input.moodEmotionLabels,
  });
}

async function retrievePersonaMemories(
  db: Database,
  input: PersonaMemoryRetrievalInput
): Promise<PersonaMemoryCandidate[]> {
  return (await retrievePersonaMemoriesWithScores(db, input)).selected;
}
function composeWorkspacePayload(input: {
  externalMemory?: PersonaExternalMemoryAudit;
  immediateMood?: PersonaAffectVector | null;
  immediateMoodUpdate?: PersonaMoodUpdateTrace | null;
  mood?: PersonaAffectVector | null;
  moodUpdate?: PersonaMoodUpdateTrace | null;
  retrieval: PersonaMemoryRetrievalPayload;
  selected: PersonaMemoryCandidate[];
  turnPlan: PersonaTurnPlan;
}): PersonaWorkspacePayload {
  const activeMemories = input.selected
    .filter((candidate) => candidate.kind === "episode")
    .map((candidate) => candidate.id);
  const activeFacts = input.selected
    .filter((candidate) => candidate.kind === "fact")
    .map((candidate) => candidate.id);
  const activeBeliefs = input.selected
    .filter((candidate) => candidate.kind === "belief")
    .map((candidate) => candidate.id);
  const activeHabits = input.selected
    .filter((candidate) => candidate.kind === "habit")
    .map((candidate) => candidate.id);
  const activeStyleProfiles = input.selected
    .filter((candidate) => candidate.kind === "style")
    .map((candidate) => candidate.id);
  const dominantEmotions = input.turnPlan.context.contextKeys.emotions.slice(
    0,
    3
  );
  const averageConfidence =
    input.selected.length === 0
      ? 0
      : input.selected.reduce(
          (sum, candidate) => sum + candidate.confidence,
          0
        ) / input.selected.length;

  return {
    activeBeliefs,
    activeHabits,
    activeFacts,
    activeMemories,
    activeStyleProfiles,
    affectiveState: {
      dominantEmotions,
      emotionLabels: input.moodUpdate?.emotionLabels ?? [],
      immediateMood: input.immediateMood ?? null,
      immediateMoodUpdate: input.immediateMoodUpdate ?? null,
      mood: input.mood ?? null,
      moodReason: {
        reasons: input.moodUpdate?.reasons ?? [],
        sources: input.moodUpdate?.sources ?? [],
      },
      moodUpdate: input.moodUpdate ?? null,
      retrievalMood: input.immediateMood ?? input.mood ?? null,
      salience:
        input.selected.reduce(
          (sum, candidate) => sum + candidate.sourceConfidence,
          0
        ) / Math.max(1, input.selected.length),
    },
    appraisalSummary: {
      answerConstraints: [
        "Do not claim direct access to private thoughts.",
        "Mention uncertainty when evidence is weak.",
        "Apply persona style only after grounding factual claims.",
      ],
      averageConfidence,
      behavioralPatternIds: activeHabits,
      relevantEvidence: input.selected.map((candidate) => ({
        confidence: candidate.confidence,
        id: candidate.id,
        // Full summaries already ship in the sectioned memory block; keep the
        // appraisal JSON (serialized into the untrimmed prompt tail) short.
        summary: truncateLine(candidate.summary, MAX_EVIDENCE_SUMMARY_CHARS),
        type: candidate.kind,
      })),
    },
    contextKeys: input.turnPlan.context.contextKeys,
    ...(input.externalMemory ? { externalMemory: input.externalMemory } : {}),
    gateOutput: input.turnPlan.gate,
    memoryRetrieval: input.retrieval,
    responsePlan: {
      answerMode: input.turnPlan.gate.answerMode,
      mustAvoid: [
        "claiming to be the real person",
        "inventing unstored private motives",
        "overconfident speculation",
      ],
      mustInclude:
        input.turnPlan.gate.safetyFlags.length > 0
          ? ["transparent safety boundary"]
          : [
              "direct answer in the persona's voice",
              "the persona's own judgment or stance",
              "evidence or uncertainty boundary",
            ],
      styleDirectives: composeStyleDirectives(input.selected),
    },
  };
}

function composeStyleDirectives(selected: PersonaMemoryCandidate[]): string[] {
  const directives = selected
    .filter(
      (candidate) => candidate.kind === "style" || candidate.kind === "habit"
    )
    .map((candidate) =>
      truncateLine(candidate.summary, MAX_EVIDENCE_SUMMARY_CHARS)
    );
  return directives.length > 0
    ? directives
    : ["use the persona's first-person voice"];
}

const MEMORY_WRITE_TRIGGER_PATTERNS = [
  /\b(remember (that|this)|please remember|don['’]t forget|forget that|correct that memory|that memory is wrong)\b/i,
  // Bare 기억해 also matches recall questions ("내 생일 기억해?"), so the write
  // trigger requires a request/imperative ending (줘, 주세요, 두, 둬, 놔, 라).
  // 마|말 covers both 기억하지 마 and the polite 기억하지 말아줘 forms.
  /(기억해\s*(줘|주|두|둬|놔|라)|기억하지\s*(마|말)|잊지\s*(마|말)|잊어\s*?(줘|버려)|그\s*기억(은|이)?\s*(틀렸|잘못))/u,
];

const MEMORY_FORGET_TRIGGER_PATTERNS = [
  /\b(forget that|don['’]t remember|do not remember|delete that memory|that memory is wrong|correct that memory)\b/i,
  /(기억하지\s*(마|말)|잊어\s*?(줘|버려)|그\s*기억(은|이)?\s*(틀렸|잘못))/u,
];

const SENSITIVE_MEMORY_PATTERNS = [
  /\b(password|passcode|api[_ -]?key|secret|private key|access token|refresh token|ssn|social security|credit card|card number|bank account)\b/i,
  /(비밀번호|암호|API\s*키|시크릿|토큰|개인키|주민등록번호|주민번호|카드번호|계좌번호|인증번호)/u,
];

export const USER_PREFERENCE_PATTERNS = [
  /\b(prefer|always|next time|from now on)\b/i,
  /(선호|항상|다음(부터|에는)|앞으로는?)/u,
];

const POST_RESPONSE_FEEDBACK_PATTERNS = [
  /\b(feedback|critique|review|judgment|reasoning|thought process|mistake|miss|wrong|improve|next time)\b/i,
  /(피드백|비판|리뷰|사고과정|판단|미스|실수|잘못|틀렸|개선|다음(부터|엔|에는))/u,
];

const POST_RESPONSE_SELF_CORRECTION_PATTERNS = [
  /\b(i agree|my mistake|i was wrong|i over[- ]?focused|i should|i should have|next time i|from now on i|i will .* first)\b/i,
  /(동의해요|제 판단 미스|제가 .*미스|너무 오래 머문|잘못 봤|놓쳤|다음(부터|엔|에는).*먼저|먼저 .*보겠습니다|우선 .*보겠습니다)/u,
];

const POST_RESPONSE_DURABLE_LESSON_PATTERNS = [
  /\b(kpi|metric|experiment|readout|guardrail|activation|retention|conversion|segment|backsolve|decision rule)\b/i,
  /(KPI|지표|실험|판정 기준|가드레일|활성화|리텐션|전환율|세그먼트|역산|의사결정 기준|우선순위)/u,
];

function matchesAnyPattern(patterns: RegExp[], text: string): boolean {
  return patterns.some((pattern) => pattern.test(text));
}

function fallbackMemoryTriageDecision(input: {
  confidence?: number;
  memoryIntent?: PersonaMemoryTriageIntent;
  memoryType?: PersonaMemoryTriageType;
  privacyLevel?: PersonaPrivacyLevel;
  reason: string;
  shouldRemember: boolean;
  source: PersonaMemoryTriageSource;
  summary?: string | null;
  themes?: string[];
}): PersonaMemoryTriageDecision {
  const summary = input.summary?.trim() || null;
  return {
    confidence: clampScore(
      input.confidence ?? (input.shouldRemember ? 0.7 : 0)
    ),
    dedupeKey: null,
    memoryIntent: input.memoryIntent ?? "none",
    memoryType:
      input.memoryType ?? (input.shouldRemember ? "interaction" : "none"),
    privacyLevel: input.privacyLevel ?? "private",
    reason: truncateLine(input.reason, 300),
    shouldRemember: input.shouldRemember,
    source: input.source,
    summary: summary ? truncateLine(summary, 500) : null,
    themes: (input.themes ?? []).slice(0, 8),
  };
}

function ruleBasedInteractionMemoryTriage(input: {
  message: string;
  selected?: PersonaMemoryCandidate[];
  source?: PersonaMemoryTriageSource;
  turnPlan: PersonaTurnPlan;
}): PersonaMemoryTriageDecision {
  const source = input.source ?? "hard_rule";
  if (matchesAnyPattern(SENSITIVE_MEMORY_PATTERNS, input.message)) {
    return fallbackMemoryTriageDecision({
      memoryIntent: "none",
      memoryType: "none",
      privacyLevel: "sensitive",
      reason:
        "The message appears to contain secret or sensitive identifier material.",
      shouldRemember: false,
      source,
      summary: null,
      themes: input.turnPlan.context.contextKeys.themes,
    });
  }

  const isExplicitMemoryRequest = matchesAnyPattern(
    MEMORY_WRITE_TRIGGER_PATTERNS,
    input.message
  );
  const isForgetRequest = matchesAnyPattern(
    MEMORY_FORGET_TRIGGER_PATTERNS,
    input.message
  );
  const queryType = input.turnPlan.gate.queryType;
  const themes = input.turnPlan.context.contextKeys.themes;
  const summary = input.message.slice(0, 500);

  if (isExplicitMemoryRequest) {
    return fallbackMemoryTriageDecision({
      confidence: 0.9,
      memoryIntent: isForgetRequest
        ? "explicit_forget_request"
        : "explicit_memory_request",
      memoryType: "interaction",
      reason: "The user explicitly requested a memory write or correction.",
      shouldRemember: true,
      source,
      summary,
      themes,
    });
  }

  if (queryType === "relationship_question") {
    return fallbackMemoryTriageDecision({
      confidence: 0.72,
      memoryIntent: "relationship_context",
      memoryType: "interaction",
      reason: "Relationship-context turns are useful for continuity.",
      shouldRemember: true,
      source,
      summary,
      themes,
    });
  }

  if (queryType === "emotional_reflection") {
    return fallbackMemoryTriageDecision({
      confidence: 0.78,
      memoryIntent: "emotional_salience",
      memoryType: "interaction",
      reason: "Emotionally reflective turns are salient continuity signals.",
      shouldRemember: true,
      source,
      summary,
      themes,
    });
  }

  if (
    NOVELTY_RECORDING_QUERY_TYPES.has(queryType) &&
    isNovelTurn(input.selected ?? [])
  ) {
    return fallbackMemoryTriageDecision({
      confidence: 0.68,
      memoryIntent:
        queryType === "autobiographical_reasoning"
          ? "autobiographical_reasoning"
          : "autobiographical_fact",
      memoryType: "interaction",
      reason:
        "The turn is autobiographical and semantically novel versus selected memory.",
      shouldRemember: true,
      source,
      summary,
      themes,
    });
  }

  return fallbackMemoryTriageDecision({
    confidence: 0,
    reason:
      "No explicit memory request, emotionally salient reflection, relationship context, or novel autobiographical content was detected by hard rules.",
    shouldRemember: false,
    source,
    summary: null,
    themes,
  });
}

type PostResponseMemoryReview = NonNullable<
  PersonaWorkspacePayload["memoryReview"]
>;

export function evaluatePostResponseInteractionMemory(input: {
  assistantMessage: string;
  turnPlan: PersonaTurnPlan;
  userMessage: string;
}): {
  memoryIntent: "self_correction_feedback" | "durable_product_lesson" | null;
  reason: string;
  shouldRemember: boolean;
  summary: string | null;
  themes: string[];
} {
  const userMessage = input.userMessage.trim();
  const assistantMessage = input.assistantMessage.trim();
  const combined = `${userMessage}\n${assistantMessage}`;
  const hasFeedbackCue = POST_RESPONSE_FEEDBACK_PATTERNS.some((pattern) =>
    pattern.test(combined)
  );
  const hasSelfCorrection = POST_RESPONSE_SELF_CORRECTION_PATTERNS.some(
    (pattern) => pattern.test(assistantMessage)
  );
  const hasDurableLesson = POST_RESPONSE_DURABLE_LESSON_PATTERNS.some(
    (pattern) => pattern.test(combined)
  );
  const queryType = input.turnPlan.gate.queryType;
  const themes = input.turnPlan.context.contextKeys.themes.slice(0, 8);

  if (
    hasFeedbackCue &&
    hasSelfCorrection &&
    (queryType === "value_judgment" ||
      queryType === "autobiographical_reasoning" ||
      queryType === "hypothetical_response" ||
      queryType === "factual_question")
  ) {
    return {
      memoryIntent: "self_correction_feedback",
      reason:
        "Assistant accepted feedback or corrected its own judgment after the response.",
      shouldRemember: true,
      summary: [
        "The persona received feedback on its reasoning and accepted a reusable correction.",
        truncateLine(assistantMessage, 420),
      ].join(" "),
      themes,
    };
  }

  if (
    hasDurableLesson &&
    /(\bshould\b|\bmust\b|\bnext time\b|\bdecision rule\b|해야|먼저|기준|원칙|다음)/i.test(
      assistantMessage
    ) &&
    (queryType === "autobiographical_reasoning" ||
      queryType === "value_judgment" ||
      queryType === "hypothetical_response")
  ) {
    return {
      memoryIntent: "durable_product_lesson",
      reason:
        "Assistant response contained a reusable product or KPI decision lesson.",
      shouldRemember: true,
      summary: truncateLine(assistantMessage, 500),
      themes,
    };
  }

  return {
    memoryIntent: null,
    reason:
      "No explicit memory request, self-correction feedback, or durable product lesson was detected after the response.",
    shouldRemember: false,
    summary: null,
    themes,
  };
}

export async function recordPostResponsePersonaMemoryReview(
  db: Database,
  input: {
    assistantMessage: string;
    assistantMessageId?: string | null;
    chatSessionId?: string | null;
    defer?: (promise: Promise<unknown>) => void;
    hindsight?: HindsightPersonaMemoryClient | null;
    logger?: PersonaLogger;
    llm?: PersonaJsonLlm;
    tenantId: string;
    persona: PersonaProfile;
    turnPlan: PersonaTurnPlan;
    userId: string;
    userMessage: string;
    workspaceId: string;
  }
): Promise<PostResponseMemoryReview> {
  const triage = await triagePostResponseInteractionMemoryWithLlm({
    assistantMessage: input.assistantMessage,
    logger: input.logger,
    llm: input.llm,
    persona: input.persona,
    turnPlan: input.turnPlan,
    userMessage: input.userMessage,
  });
  let memoryId: string | null = null;

  if (triage.shouldRemember && triage.summary) {
    const [interaction] = await db
      .insert(personaInteractionMemories)
      .values({
        chatMessageId: input.assistantMessageId,
        chatSessionId: input.chatSessionId,
        interactionSummary: triage.summary.slice(0, 500),
        newPersonaMemory: {
          confidence: triage.confidence,
          dedupeKey: triage.dedupeKey,
          memoryIntent: triage.memoryIntent,
          memoryType: triage.memoryType,
          queryType: input.turnPlan.gate.queryType,
          reason: triage.reason,
          source: triage.source,
          themes: triage.themes,
        },
        tenantId: input.tenantId,
        personaId: input.persona.id,
        shouldConsolidate: true,
        state: "consolidation_candidate",
        ttlDays: 30,
        userId: input.userId,
      })
      .returning({ id: personaInteractionMemories.id });
    memoryId = interaction?.id ?? null;
    if (memoryId) {
      scheduleHindsightInteractionRetain({
        chatMessageId: input.assistantMessageId,
        chatSessionId: input.chatSessionId,
        content: triage.summary,
        db,
        defer: input.defer,
        documentId: `persona_interaction_${memoryId}`,
        hindsight: input.hindsight,
        logger: input.logger,
        localTargetId: memoryId,
        tenantId: input.tenantId,
        persona: input.persona,
        themes: triage.themes,
        userId: input.userId,
      });
    }
  }

  const review: PostResponseMemoryReview = {
    createdAt: new Date().toISOString(),
    memoryId,
    memoryIntent: triage.memoryIntent === "none" ? null : triage.memoryIntent,
    reason: triage.reason,
    shouldRemember: triage.shouldRemember,
    status: memoryId ? "recorded" : "skipped",
    version: 1,
  };

  const [workspace] = await db
    .select({ payload: personaWorkspaceStates.payload })
    .from(personaWorkspaceStates)
    .where(eq(personaWorkspaceStates.id, input.workspaceId))
    .limit(1);
  if (workspace) {
    await db
      .update(personaWorkspaceStates)
      .set({
        payload: {
          ...workspace.payload,
          memoryReview: review,
        },
      })
      .where(eq(personaWorkspaceStates.id, input.workspaceId));
  }

  return review;
}

// Personal query types where a turn that matches no stored memory is worth
// remembering: the user is sharing or probing identity-relevant content the
// persona has never seen (novelty gating, flashbulb-style auto-encoding).
const NOVELTY_RECORDING_QUERY_TYPES: ReadonlySet<PersonaQueryType> = new Set([
  "autobiographical_fact",
  "autobiographical_reasoning",
  "emotional_reflection",
]);
// Raw (1 + cos) / 2 similarity below which the turn counts as novel.
const NOVELTY_SIMILARITY_THRESHOLD = 0.62;

function isNovelTurn(selected: PersonaMemoryCandidate[]): boolean {
  const similarities = selected
    .filter(
      (candidate) =>
        candidate.kind === "episode" ||
        candidate.kind === "fact" ||
        candidate.kind === "source"
    )
    .map((candidate) => candidate.semanticSimilarity)
    .filter((value): value is number => value !== null);
  // Without embeddings there is no reliable novelty signal; stay conservative.
  if (similarities.length === 0) {
    return false;
  }
  return Math.max(...similarities) < NOVELTY_SIMILARITY_THRESHOLD;
}

function formatSelectedMemoryForTriage(
  selected: PersonaMemoryCandidate[] | undefined
): string[] {
  const memories = (selected ?? []).slice(0, MEMORY_TRIAGE_SELECTED_MEMORY_CAP);
  if (memories.length === 0) {
    return ["- none"];
  }
  return memories.map((candidate) => {
    const similarity =
      candidate.semanticSimilarity === null
        ? "n/a"
        : candidate.semanticSimilarity.toFixed(2);
    return `- ${candidate.kind}:${candidate.id} similarity=${similarity} confidence=${candidate.confidence.toFixed(2)} ${truncateLine(candidate.summary, 180)}`;
  });
}

function buildMemoryTriageUserPrompt(input: {
  assistantMessage?: string | null;
  message: string;
  persona: PersonaProfile;
  selected?: PersonaMemoryCandidate[];
  turnPlan: PersonaTurnPlan;
}): string {
  return [
    `Persona: ${input.persona.displayName} (${input.persona.personaKey})`,
    `Turn queryType: ${input.turnPlan.gate.queryType}`,
    `Safety flags: ${input.turnPlan.gate.safetyFlags.join(", ") || "none"}`,
    `Needed memory types: ${input.turnPlan.gate.neededMemoryTypes.join(", ") || "none"}`,
    `Themes: ${input.turnPlan.context.contextKeys.themes.join(", ") || "none"}`,
    `Entities: ${input.turnPlan.context.contextKeys.entities.join(", ") || "none"}`,
    "",
    "Selected memories already active for this turn:",
    ...formatSelectedMemoryForTriage(input.selected),
    "",
    "User message:",
    truncateLine(input.message, 1_200),
    ...(input.assistantMessage
      ? ["", "Assistant response:", truncateLine(input.assistantMessage, 1_400)]
      : []),
  ].join("\n");
}

function applyMemoryTriageHardGates(input: {
  decision: Omit<PersonaMemoryTriageDecision, "source">;
  hardRuleDecision: PersonaMemoryTriageDecision;
  message: string;
  source: PersonaMemoryTriageSource;
  turnPlan: PersonaTurnPlan;
}): PersonaMemoryTriageDecision {
  const decision: PersonaMemoryTriageDecision = {
    ...input.decision,
    source: input.source,
  };
  const hasSensitiveCue =
    matchesAnyPattern(SENSITIVE_MEMORY_PATTERNS, input.message) ||
    decision.privacyLevel === "sensitive";
  if (hasSensitiveCue) {
    return fallbackMemoryTriageDecision({
      confidence: Math.max(
        decision.confidence,
        input.hardRuleDecision.confidence
      ),
      memoryIntent: "none",
      memoryType: "none",
      privacyLevel: "sensitive",
      reason:
        "Memory triage skipped retention because the turn contains or was classified as sensitive.",
      shouldRemember: false,
      source: input.source,
      summary: null,
      themes: decision.themes,
    });
  }

  const explicitHardRule = input.hardRuleDecision.shouldRemember;
  if (
    input.turnPlan.gate.safetyFlags.length > 0 &&
    !explicitHardRule &&
    decision.shouldRemember
  ) {
    return fallbackMemoryTriageDecision({
      confidence: decision.confidence,
      memoryIntent: "none",
      memoryType: "none",
      reason: "Memory triage skipped retention while safety flags are present.",
      shouldRemember: false,
      source: input.source,
      summary: null,
      themes: decision.themes,
    });
  }

  if (!decision.shouldRemember && explicitHardRule) {
    return {
      ...input.hardRuleDecision,
      reason: `${input.hardRuleDecision.reason} LLM triage did not produce a stronger durable candidate: ${decision.reason}`,
      source: input.hardRuleDecision.source,
    };
  }

  if (
    !decision.shouldRemember ||
    decision.memoryIntent === "none" ||
    decision.memoryType === "none" ||
    !decision.summary
  ) {
    return fallbackMemoryTriageDecision({
      confidence: decision.confidence,
      memoryIntent: decision.memoryIntent,
      memoryType: decision.memoryType,
      privacyLevel: decision.privacyLevel,
      reason: decision.reason,
      shouldRemember: false,
      source: input.source,
      summary: decision.summary,
      themes: decision.themes,
    });
  }

  if (decision.confidence < MEMORY_TRIAGE_CONFIDENCE_THRESHOLD) {
    if (explicitHardRule) {
      return {
        ...input.hardRuleDecision,
        reason: `${input.hardRuleDecision.reason} LLM confidence was below automatic triage threshold (${decision.confidence.toFixed(2)}).`,
        source: input.hardRuleDecision.source,
      };
    }
    return fallbackMemoryTriageDecision({
      confidence: decision.confidence,
      memoryIntent: decision.memoryIntent,
      memoryType: decision.memoryType,
      privacyLevel: decision.privacyLevel,
      reason: `LLM confidence ${decision.confidence.toFixed(2)} is below the ${MEMORY_TRIAGE_CONFIDENCE_THRESHOLD.toFixed(2)} retention threshold.`,
      shouldRemember: false,
      source: input.source,
      summary: decision.summary,
      themes: decision.themes,
    });
  }

  return decision;
}

export async function triageInteractionMemoryWithLlm(input: {
  logger?: PersonaLogger;
  llm?: PersonaJsonLlm;
  message: string;
  persona: PersonaProfile;
  selected?: PersonaMemoryCandidate[];
  turnPlan: PersonaTurnPlan;
}): Promise<PersonaMemoryTriageDecision> {
  const hardRuleDecision = ruleBasedInteractionMemoryTriage(input);
  if (!input.llm) {
    return hardRuleDecision;
  }
  const logger = resolvePersonaLogger(input.logger);
  try {
    const output = await input.llm({
      systemPrompt: PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT,
      userPrompt: buildMemoryTriageUserPrompt(input),
    });
    const decision = parsePersonaMemoryTriageDecision(output);
    return applyMemoryTriageHardGates({
      decision,
      hardRuleDecision,
      message: input.message,
      source: "llm",
      turnPlan: input.turnPlan,
    });
  } catch (error) {
    logger.warn(
      personaLogMessage(
        "[persona-memory] interaction memory triage failed:",
        error
      )
    );
    return {
      ...ruleBasedInteractionMemoryTriage({
        ...input,
        source: "llm_fallback",
      }),
      source: "llm_fallback",
    };
  }
}

export async function triagePostResponseInteractionMemoryWithLlm(input: {
  assistantMessage: string;
  logger?: PersonaLogger;
  llm?: PersonaJsonLlm;
  persona: PersonaProfile;
  turnPlan: PersonaTurnPlan;
  userMessage: string;
}): Promise<PersonaMemoryTriageDecision> {
  const hardEvaluation = evaluatePostResponseInteractionMemory(input);
  const hardRuleDecision = fallbackMemoryTriageDecision({
    confidence: hardEvaluation.shouldRemember ? 0.78 : 0,
    memoryIntent: hardEvaluation.memoryIntent ?? "none",
    memoryType:
      hardEvaluation.memoryIntent === "self_correction_feedback"
        ? "correction"
        : hardEvaluation.memoryIntent === "durable_product_lesson"
          ? "lesson"
          : "none",
    reason: hardEvaluation.reason,
    shouldRemember: hardEvaluation.shouldRemember,
    source: "hard_rule",
    summary: hardEvaluation.summary,
    themes: hardEvaluation.themes,
  });
  if (!input.llm) {
    return hardRuleDecision;
  }
  const logger = resolvePersonaLogger(input.logger);
  try {
    const output = await input.llm({
      systemPrompt: PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT,
      userPrompt: buildMemoryTriageUserPrompt({
        assistantMessage: input.assistantMessage,
        message: input.userMessage,
        persona: input.persona,
        turnPlan: input.turnPlan,
      }),
    });
    const decision = parsePersonaMemoryTriageDecision(output);
    return applyMemoryTriageHardGates({
      decision,
      hardRuleDecision,
      message: `${input.userMessage}\n${input.assistantMessage}`,
      source: "llm",
      turnPlan: input.turnPlan,
    });
  } catch (error) {
    logger.warn(
      personaLogMessage(
        "[persona-memory] post-response memory triage failed:",
        error
      )
    );
    return {
      ...hardRuleDecision,
      source: "llm_fallback",
    };
  }
}

function scheduleHindsightInteractionRetain(input: {
  chatMessageId?: string | null;
  chatSessionId?: string | null;
  content: string;
  db: Database;
  defer?: (promise: Promise<unknown>) => void;
  documentId: string;
  hindsight?: HindsightPersonaMemoryClient | null;
  logger?: PersonaLogger;
  localTargetId: string;
  tenantId: string;
  persona: PersonaProfile;
  themes: string[];
  userId: string;
}): void {
  scheduleHindsightRetain({
    chatMessageId: input.chatMessageId,
    chatSessionId: input.chatSessionId,
    db: input.db,
    defer: input.defer,
    hindsight: input.hindsight,
    logger: input.logger,
    localTargetId: input.localTargetId,
    localTargetKind: "interaction",
    metadata: { documentId: input.documentId },
    tenantId: input.tenantId,
    personaId: input.persona.id,
    privacyLevel: "private",
    retainInput: {
      chatMessageId: input.chatMessageId,
      chatSessionId: input.chatSessionId,
      content: input.content,
      context: "Persona-user interaction memory",
      documentId: input.documentId,
      tenantId: input.tenantId,
      personaId: input.persona.id,
      personaKey: input.persona.personaKey,
      privacyLevel: "private",
      scope: "persona_user",
      themes: input.themes,
      timestamp: new Date(),
      userId: input.userId,
    },
    userId: input.userId,
  });
}

export function shouldRecordInteractionMemory(input: {
  message: string;
  selected?: PersonaMemoryCandidate[];
  turnPlan: PersonaTurnPlan;
}): boolean {
  return ruleBasedInteractionMemoryTriage(input).shouldRemember;
}

async function maybeRecordInteractionMemory(
  db: Database,
  input: {
    chatMessageId?: string | null;
    chatSessionId?: string | null;
    defer?: (promise: Promise<unknown>) => void;
    hindsight?: HindsightPersonaMemoryClient | null;
    logger?: PersonaLogger;
    llm?: PersonaJsonLlm;
    message: string;
    tenantId: string;
    persona: PersonaProfile;
    selected?: PersonaMemoryCandidate[];
    turnPlan: PersonaTurnPlan;
    userId: string;
  }
): Promise<void> {
  const triage = await triageInteractionMemoryWithLlm({
    logger: input.logger,
    llm: input.llm,
    message: input.message,
    persona: input.persona,
    selected: input.selected,
    turnPlan: input.turnPlan,
  });
  if (!triage.shouldRemember || !triage.summary) {
    return;
  }

  const interaction = {
    chatMessageId: input.chatMessageId,
    chatSessionId: input.chatSessionId,
    interactionSummary: triage.summary.slice(0, 500),
    newPersonaMemory: {
      confidence: triage.confidence,
      dedupeKey: triage.dedupeKey,
      memoryIntent: triage.memoryIntent,
      memoryType: triage.memoryType,
      queryType: input.turnPlan.gate.queryType,
      reason: triage.reason,
      source: triage.source,
      themes: triage.themes,
    },
    newUserPreference:
      triage.memoryIntent === "user_preference" ||
      triage.memoryType === "preference" ||
      USER_PREFERENCE_PATTERNS.some((pattern) => pattern.test(input.message))
        ? input.message.slice(0, 280)
        : null,
    tenantId: input.tenantId,
    personaId: input.persona.id,
    shouldConsolidate: true,
    state: "consolidation_candidate",
    ttlDays: 30,
    userId: input.userId,
  } satisfies NewPersonaInteractionMemory;

  const [inserted] = await db
    .insert(personaInteractionMemories)
    .values(interaction)
    .returning({ id: personaInteractionMemories.id });
  if (!inserted) {
    return;
  }

  if (triage.memoryIntent !== "explicit_forget_request") {
    scheduleHindsightInteractionRetain({
      chatMessageId: input.chatMessageId,
      chatSessionId: input.chatSessionId,
      content: interaction.interactionSummary,
      db,
      defer: input.defer,
      documentId: `persona_interaction_${inserted.id}`,
      hindsight: input.hindsight,
      logger: input.logger,
      localTargetId: inserted.id,
      tenantId: input.tenantId,
      persona: input.persona,
      themes: triage.themes,
      userId: input.userId,
    });
  }
}

export async function preparePersonaRuntimeContext(
  db: Database,
  input: {
    chatMessageId?: string | null;
    chatSessionId?: string | null;
    defer?: (promise: Promise<unknown>) => void;
    disclosurePolicy?: DisclosurePolicy;
    embed?: PersonaEmbedder;
    hindsight?: HindsightPersonaMemoryClient | null;
    logger?: PersonaLogger;
    llm: PersonaJsonLlm;
    message: string;
    tenantId: string;
    personaKey: string;
    userId: string;
  }
): Promise<PersonaRuntimeContext> {
  const persona = await loadPersonaProfile(db, {
    tenantId: input.tenantId,
    personaKey: input.personaKey,
  });
  const disclosurePolicy = resolveDisclosurePolicy(input.disclosurePolicy);
  const logger = resolvePersonaLogger(input.logger);
  const now = new Date();
  const turnPlan = await planPersonaTurnWithLlm({
    llm: input.llm,
    message: input.message,
    persona,
  });
  // Current-turn cues should color recall immediately: first decay the stored
  // mood, then apply only user-message/query cues before memory retrieval.
  const storedMood = await loadPersonaMoodState(db, {
    tenantId: input.tenantId,
    personaId: persona.id,
    userId: input.userId,
  });
  const storedMoodVector = storedMood
    ? {
        arousal: storedMood.arousal,
        dominance: storedMood.dominance,
        valence: storedMood.valence,
      }
    : null;
  const elapsedMs = storedMood
    ? now.getTime() - storedMood.updatedAt.getTime()
    : 0;
  const immediateTurnAffect = explainTurnAffect({ selected: [], turnPlan });
  const immediateMoodUpdate = calculatePersonaMoodUpdate({
    current: storedMoodVector,
    elapsedMs,
    impulse: immediateTurnAffect.impulse,
    turnAffect: immediateTurnAffect,
  });
  const immediateMood = immediateMoodUpdate.mood;
  const memorySelection = await retrievePersonaMemoriesWithScores(db, {
    context: turnPlan.context,
    embed: input.embed,
    logger,
    mood: immediateMood,
    moodEmotionLabels: immediateMoodUpdate.trace.emotionLabels,
    tenantId: input.tenantId,
    persona,
  });
  const selected = memorySelection.selected;
  const externalMemory = await maybeRecallHindsightPersonaMemory({
    client: input.hindsight,
    localSelected: selected,
    message: input.message,
    tenantId: input.tenantId,
    persona,
    turnPlan,
    userId: input.userId,
  });
  // The turn's emotional impulse (activated memories + query cues) moves the
  // mood, which is persisted for the next turn and shown to the agent.
  const turnAffect = explainTurnAffect({ selected, turnPlan });
  const moodUpdate = calculatePersonaMoodUpdate({
    current: storedMoodVector,
    elapsedMs,
    impulse: turnAffect.impulse,
    turnAffect,
  });
  const mood = moodUpdate.mood;
  try {
    await persistPersonaMoodState(db, {
      mood,
      tenantId: input.tenantId,
      personaId: persona.id,
      turnCount: (storedMood?.turnCount ?? 0) + 1,
      userId: input.userId,
    });
  } catch (error) {
    logger.warn(
      personaLogMessage("[persona-memory] mood persistence failed:", error)
    );
  }
  const workspacePayload = composeWorkspacePayload({
    externalMemory,
    immediateMood,
    immediateMoodUpdate: immediateMoodUpdate.trace,
    mood,
    moodUpdate: moodUpdate.trace,
    retrieval: memorySelection.retrieval,
    selected,
    turnPlan,
  });

  const [workspace] = await db
    .insert(personaWorkspaceStates)
    .values({
      chatMessageId: input.chatMessageId,
      chatSessionId: input.chatSessionId,
      tenantId: input.tenantId,
      payload: workspacePayload,
      personaId: persona.id,
      userId: input.userId,
      userMessage: input.message,
    })
    .returning();
  if (!workspace) {
    throw new Error("Failed to persist persona workspace state.");
  }

  await maybeRecordInteractionMemory(db, {
    chatMessageId: input.chatMessageId,
    chatSessionId: input.chatSessionId,
    message: input.message,
    tenantId: input.tenantId,
    persona,
    defer: input.defer,
    hindsight: input.hindsight,
    logger,
    llm: input.llm,
    selected,
    turnPlan,
    userId: input.userId,
  });

  // Use-dependent strengthening: recalling a memory makes it easier to recall
  // next time. Best-effort; a failed bump must not break the turn.
  try {
    await markPersonaMemoriesActivated(db, { selected });
  } catch (error) {
    logger.warn(
      personaLogMessage("[persona-memory] activation bump failed:", error)
    );
  }

  return {
    disclosurePolicy,
    persona,
    promptContext: formatWorkspacePrompt({
      disclosurePolicy,
      mood,
      moodUpdate: moodUpdate.trace,
      profile: persona,
      selected,
      turnPlan,
      workspacePayload,
    }),
    turnPlan,
    workspaceId: workspace.id,
  };
}

async function markPersonaMemoriesActivated(
  db: Database,
  input: { now?: Date; selected: PersonaMemoryCandidate[] }
): Promise<void> {
  const now = input.now ?? new Date();
  const idsOfKind = (kind: PersonaMemoryCandidate["kind"]) =>
    input.selected
      .filter((candidate) => candidate.kind === kind)
      .map((candidate) => candidate.id);

  const episodeIds = idsOfKind("episode");
  const beliefIds = idsOfKind("belief");
  const factIds = idsOfKind("fact");
  const habitIds = idsOfKind("habit");
  const styleIds = idsOfKind("style");

  await Promise.all([
    episodeIds.length > 0
      ? db
          .update(personaEpisodeMemories)
          .set({
            activationCount: sql`${personaEpisodeMemories.activationCount} + 1`,
            lastActivatedAt: now,
          })
          .where(inArray(personaEpisodeMemories.id, episodeIds))
      : Promise.resolve(),
    beliefIds.length > 0
      ? db
          .update(personaSemanticBeliefs)
          .set({
            activationCount: sql`${personaSemanticBeliefs.activationCount} + 1`,
            lastActivatedAt: now,
          })
          .where(inArray(personaSemanticBeliefs.id, beliefIds))
      : Promise.resolve(),
    factIds.length > 0
      ? db
          .update(personaFacts)
          .set({
            activationCount: sql`${personaFacts.activationCount} + 1`,
            lastActivatedAt: now,
          })
          .where(inArray(personaFacts.id, factIds))
      : Promise.resolve(),
    habitIds.length > 0
      ? db
          .update(personaHabitPatterns)
          .set({
            activationCount: sql`${personaHabitPatterns.activationCount} + 1`,
            lastActivatedAt: now,
          })
          .where(inArray(personaHabitPatterns.id, habitIds))
      : Promise.resolve(),
    styleIds.length > 0
      ? db
          .update(personaStyleProfiles)
          .set({
            activationCount: sql`${personaStyleProfiles.activationCount} + 1`,
            lastActivatedAt: now,
          })
          .where(inArray(personaStyleProfiles.id, styleIds))
      : Promise.resolve(),
  ]);
}
