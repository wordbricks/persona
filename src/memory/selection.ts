import type {
  PersonaEmotionAnnotation,
  PersonaMemoryRetrievalPayload,
  PersonaPrivacyLevel,
} from "../schema";
import { clampScore, emotionLabelFamily } from "./mood";
import { MEMORY_KIND_BUDGETS } from "./types";
import type {
  PersonaAffectVector,
  PersonaContextGatewayOutput,
  PersonaMemoryActivationScoreBreakdown,
  PersonaMemoryCandidate,
  PersonaMemoryRankedEntry,
  PersonaMemorySelectionResult,
  PersonaMoodEmotionLabel,
} from "./types";

export function tokenize(input: string): Set<string> {
  const tokens = new Set<string>();
  for (const token of input
    .toLowerCase()
    .split(/[^a-z0-9가-힣_]+/iu)
    .filter((entry) => entry.length >= 2)) {
    tokens.add(token);
    if (/^[가-힣]{3,}$/u.test(token)) {
      const stripped = token.replace(
        /(께서는|에서는|에게는|으로는|로는|에게|에서|으로|로서|보다|처럼|까지|부터|은|는|이|가|을|를|에|의|도|만|과|와|랑|로)$/u,
        ""
      );
      if (stripped.length >= 2) {
        tokens.add(stripped);
      }
    }
  }
  return tokens;
}

export function overlapScore(
  left: Iterable<string>,
  right: Iterable<string>
): number {
  const leftSet = new Set(left);
  const rightSet = new Set(right);
  if (leftSet.size === 0) {
    return 0.5;
  }
  let matches = 0;
  for (const value of leftSet) {
    if (rightSet.has(value)) {
      matches += 1;
    }
  }
  return clampScore(matches / leftSet.size);
}

// text-embedding-3 cosine similarity mapped through (1 + cos) / 2 sits near
// 0.55 for unrelated text. Rescale 0.55 -> 0 and 0.90 -> 1 so embedding
// similarity is comparable with lexical overlap scores in the same weighted
// sum.
export function normalizeCosineSimilarity(value: number | null): number {
  if (value === null || !Number.isFinite(value)) {
    return 0;
  }
  return clampScore((value - 0.55) / 0.35);
}

export const MS_PER_DAY = 86_400_000;
const EPISODIC_HALF_LIFE_DAYS = 90;
const RECALL_PRIMING_HALF_LIFE_DAYS = 30;
// Forgetting lowers availability but never erases it: a strong enough cue can
// still surface an old memory, it just no longer wins ties.
const MEMORY_AVAILABILITY_FLOOR = 0.35;

// How easily a memory comes to mind right now. Models the forgetting curve
// (exponential decay from creation), amygdala-modulated retention (emotionally
// salient episodes decay slower), recall priming (recently activated memories
// are temporarily more available), and use-dependent strengthening (memories
// recalled often stay available).
export function calculateMemoryAvailability(input: {
  candidate: Pick<
    PersonaMemoryCandidate,
    | "activationCount"
    | "createdAt"
    | "kind"
    | "lastActivatedAt"
    | "sourceConfidence"
  >;
  now: Date;
}): number {
  const { candidate } = input;
  // Identity memory (beliefs, habits, style) is tonic; personality does not
  // fade on conversational timescales.
  if (
    candidate.kind === "belief" ||
    candidate.kind === "habit" ||
    candidate.kind === "style"
  ) {
    return 1;
  }
  const ageDays = Math.max(
    0,
    (input.now.getTime() - candidate.createdAt.getTime()) / MS_PER_DAY
  );
  const halfLifeDays =
    EPISODIC_HALF_LIFE_DAYS * (1 + clampScore(candidate.sourceConfidence));
  const creationDecay = 0.5 ** (ageDays / halfLifeDays);
  const recallAgeDays =
    candidate.lastActivatedAt === null
      ? null
      : Math.max(
          0,
          (input.now.getTime() - candidate.lastActivatedAt.getTime()) /
            MS_PER_DAY
        );
  const priming =
    recallAgeDays === null
      ? 0
      : 0.5 ** (recallAgeDays / RECALL_PRIMING_HALF_LIFE_DAYS);
  const usage = Math.min(
    1,
    Math.log1p(candidate.activationCount) / Math.log(20)
  );
  const trace = Math.max(
    creationDecay,
    clampScore(0.6 * priming + 0.4 * usage)
  );
  return MEMORY_AVAILABILITY_FLOOR + (1 - MEMORY_AVAILABILITY_FLOOR) * trace;
}

function isRetrievablePrivacy(value: PersonaPrivacyLevel): boolean {
  return value === "public" || value === "internal" || value === "private";
}

const MOOD_VALENCE_CONGRUENCE_WEIGHT = 0.6;
const MOOD_AROUSAL_CONGRUENCE_WEIGHT = 0.25;
const MOOD_DOMINANCE_CONGRUENCE_WEIGHT = 0.15;
const MOOD_AFFECTIVE_MULTIPLIER_FLOOR = 0.8;
const MOOD_PAD_MULTIPLIER_SPAN = 0.35;
const MOOD_EMOTION_LABEL_MULTIPLIER_SPAN = 0.05;

function calculateMoodAffectCongruence(input: {
  affect: PersonaAffectVector | null;
  mood?: PersonaAffectVector | null;
}): number {
  if (!input.mood || !input.affect) {
    return 0.5;
  }
  const padDistance =
    MOOD_VALENCE_CONGRUENCE_WEIGHT *
      (Math.abs(input.mood.valence - input.affect.valence) / 2) +
    MOOD_AROUSAL_CONGRUENCE_WEIGHT *
      Math.abs(input.mood.arousal - input.affect.arousal) +
    MOOD_DOMINANCE_CONGRUENCE_WEIGHT *
      Math.abs(input.mood.dominance - input.affect.dominance);
  return 1 - clampScore(padDistance);
}

function calculateEmotionLabelOverlap(input: {
  candidateEmotions: PersonaEmotionAnnotation[];
  moodEmotionLabels?: PersonaMoodEmotionLabel[] | null;
}): number {
  if (
    !input.moodEmotionLabels ||
    input.moodEmotionLabels.length === 0 ||
    input.candidateEmotions.length === 0
  ) {
    return 0;
  }
  const moodIntensityByFamily = new Map<string, number>();
  for (const label of input.moodEmotionLabels) {
    const family = emotionLabelFamily(label.label);
    moodIntensityByFamily.set(
      family,
      Math.max(moodIntensityByFamily.get(family) ?? 0, label.intensity)
    );
  }
  return input.candidateEmotions.reduce((best, emotion) => {
    const moodIntensity = moodIntensityByFamily.get(
      emotionLabelFamily(emotion.emotion)
    );
    if (moodIntensity === undefined) {
      return best;
    }
    return Math.max(
      best,
      (clampScore(moodIntensity) + clampScore(emotion.intensity)) / 2
    );
  }, 0);
}

function calculateMemoryActivationScore(input: {
  candidate: PersonaMemoryCandidate;
  context: PersonaContextGatewayOutput;
  mood?: PersonaAffectVector | null;
  moodEmotionLabels?: PersonaMoodEmotionLabel[] | null;
  query: string;
}): PersonaMemoryActivationScoreBreakdown {
  const queryTokens = tokenize(input.query);
  const candidateTokens = tokenize(input.candidate.text);
  // Hybrid retrieval: lexical overlap catches verbatim proper nouns, the
  // embedding similarity catches paraphrases and cross-language matches.
  const lexicalSimilarity = overlapScore(queryTokens, candidateTokens);
  const normalizedSemanticSimilarity = normalizeCosineSimilarity(
    input.candidate.semanticSimilarity
  );
  const semanticSimilarity = Math.max(
    lexicalSimilarity,
    normalizedSemanticSimilarity
  );
  const entityMatch = overlapScore(
    input.context.contextKeys.entities,
    candidateTokens
  );
  const themeMatch = overlapScore(
    input.context.contextKeys.themes,
    input.candidate.themes
  );
  const temporalMatch = overlapScore(
    input.context.contextKeys.timePeriods,
    candidateTokens
  );
  // Mood-congruent recall: episodes whose stored affect matches the immediate
  // mood come to mind more easily. Valence remains dominant; arousal and
  // dominance are secondary, and exact emotion-label overlap gets only a small
  // bounded bonus so mood cannot override relevance.
  const moodCongruence = calculateMoodAffectCongruence({
    affect: input.candidate.affect,
    mood: input.mood,
  });
  const emotionLabelOverlap = calculateEmotionLabelOverlap({
    candidateEmotions: input.candidate.emotions,
    moodEmotionLabels: input.moodEmotionLabels,
  });
  const affectiveSalience =
    input.candidate.kind === "episode"
      ? clampScore(
          input.candidate.sourceConfidence *
            input.candidate.retrievalBoost *
            (MOOD_AFFECTIVE_MULTIPLIER_FLOOR +
              MOOD_PAD_MULTIPLIER_SPAN * moodCongruence +
              MOOD_EMOTION_LABEL_MULTIPLIER_SPAN * emotionLabelOverlap)
        )
      : 0.2;
  const identityRelevance = input.candidate.strength;
  const sourceConfidence = input.candidate.sourceConfidence;
  const recencyOrLifeStageRelevance = 0.5;
  const privacyOrPolicyRisk = isRetrievablePrivacy(input.candidate.privacyLevel)
    ? 0
    : 1;

  const components = {
    affectiveSalience,
    emotionLabelOverlap,
    entityMatch,
    identityRelevance,
    lexicalSimilarity,
    moodCongruence,
    normalizedSemanticSimilarity,
    privacyOrPolicyRisk,
    recencyOrLifeStageRelevance,
    semanticSimilarity,
    sourceConfidence,
    temporalMatch,
    themeMatch,
  };

  const score = clampScore(
    0.25 * semanticSimilarity +
      0.15 * entityMatch +
      0.15 * themeMatch +
      0.1 * temporalMatch +
      0.15 * affectiveSalience +
      0.1 * identityRelevance +
      0.05 * sourceConfidence +
      0.05 * recencyOrLifeStageRelevance -
      0.2 * privacyOrPolicyRisk
  );

  return { components, score };
}
// One-hop spreading activation over explicit memory links (belief ->
// supporting episode/source, episode -> source). When one memory fires, the
// memories wired to it get a fraction of its activation, mimicking
// associative recall: remembering a belief pulls up the episode that taught
// it. Boosts use max (not sum) so densely linked graphs cannot snowball, and
// a single pass keeps activation local to direct associations.
const SPREADING_ACTIVATION_FACTOR = 0.3;

function applySpreadingActivation(ranked: PersonaMemoryRankedEntry[]): void {
  const indexByAlias = new Map<string, number[]>();
  ranked.forEach((entry, index) => {
    for (const alias of [entry.candidate.id, ...entry.candidate.aliasIds]) {
      const bucket = indexByAlias.get(alias);
      if (bucket) {
        bucket.push(index);
      } else {
        indexByAlias.set(alias, [index]);
      }
    }
  });

  const boosts = new Array<number>(ranked.length).fill(0);
  ranked.forEach((entry, index) => {
    for (const linkId of entry.candidate.linkIds) {
      for (const linkedIndex of indexByAlias.get(linkId) ?? []) {
        if (linkedIndex === index) {
          continue;
        }
        // Associations work both ways: the linker activates the linked memory
        // and a strongly active linked memory re-activates its linker.
        boosts[linkedIndex] = Math.max(
          boosts[linkedIndex] ?? 0,
          SPREADING_ACTIVATION_FACTOR * entry.rank
        );
        boosts[index] = Math.max(
          boosts[index] ?? 0,
          SPREADING_ACTIVATION_FACTOR * (ranked[linkedIndex]?.rank ?? 0)
        );
      }
    }
  });

  ranked.forEach((entry, index) => {
    const nextRank = clampScore(entry.rank + (boosts[index] ?? 0));
    entry.spreadingBoost = clampScore(nextRank - entry.rank);
    entry.rank = nextRank;
  });
}

export function selectPersonaMemories(input: {
  candidates: PersonaMemoryCandidate[];
  context: PersonaContextGatewayOutput;
  mood?: PersonaAffectVector | null;
  moodEmotionLabels?: PersonaMoodEmotionLabel[] | null;
  now?: Date;
}): PersonaMemoryCandidate[] {
  return selectPersonaMemoriesWithScores(input).selected;
}

function memoryScoreKey(
  candidate: Pick<PersonaMemoryCandidate, "id" | "kind">
) {
  return `${candidate.kind}:${candidate.id}`;
}

function serializeMemoryRetrievalCandidate(input: {
  entry: PersonaMemoryRankedEntry;
  selectedOrder: number | null;
}): PersonaMemoryRetrievalPayload["candidates"][number] {
  const { candidate } = input.entry;
  return {
    activationCount: candidate.activationCount,
    activationScore: input.entry.activationScore,
    availability: input.entry.availability,
    baseRank: input.entry.baseRank,
    components: input.entry.components,
    confidence: candidate.confidence,
    createdAt: candidate.createdAt.toISOString(),
    excludedReason: input.entry.excludedReason,
    id: candidate.id,
    kind: candidate.kind,
    lastActivatedAt: candidate.lastActivatedAt?.toISOString() ?? null,
    privacyLevel: candidate.privacyLevel,
    query: input.entry.query,
    rank: input.entry.rank,
    rankBeforeSpreading: input.entry.rankBeforeSpreading,
    retrievalBoost: candidate.retrievalBoost,
    selected: input.selectedOrder !== null,
    selectedOrder: input.selectedOrder,
    semanticSimilarity: candidate.semanticSimilarity,
    sourceConfidence: candidate.sourceConfidence,
    spreadingBoost: input.entry.spreadingBoost,
    strength: candidate.strength,
  };
}

export function selectPersonaMemoriesWithScores(input: {
  candidates: PersonaMemoryCandidate[];
  context: PersonaContextGatewayOutput;
  mood?: PersonaAffectVector | null;
  moodEmotionLabels?: PersonaMoodEmotionLabel[] | null;
  now?: Date;
}): PersonaMemorySelectionResult {
  const now = input.now ?? new Date();
  const queryByKind = (kind: PersonaMemoryCandidate["kind"]) => {
    if (kind === "episode") {
      return input.context.retrievalQueries.episodic;
    }
    if (kind === "belief") {
      return input.context.retrievalQueries.semantic;
    }
    if (kind === "fact") {
      return input.context.retrievalQueries.source;
    }
    if (kind === "habit") {
      return input.context.retrievalQueries.habit;
    }
    if (kind === "style") {
      return input.context.retrievalQueries.style;
    }
    return input.context.retrievalQueries.source;
  };

  const scored = input.candidates.map((candidate): PersonaMemoryRankedEntry => {
    const query = queryByKind(candidate.kind);
    if (!isRetrievablePrivacy(candidate.privacyLevel)) {
      return {
        activationScore: null,
        availability: null,
        baseRank: null,
        candidate,
        components: null,
        excludedReason: "privacy",
        query,
        rank: 0,
        rankBeforeSpreading: null,
        spreadingBoost: 0,
      };
    }

    const activation = calculateMemoryActivationScore({
      candidate,
      context: input.context,
      mood: input.mood,
      moodEmotionLabels: input.moodEmotionLabels,
      query,
    });
    // Beliefs and habits rank partly on conviction strength so strong
    // identity memory survives weak lexical overlap with the query. Facts use
    // a smaller confidence term so verified concrete details can surface
    // without overwhelming source/query relevance.
    const baseRank =
      candidate.kind === "belief" || candidate.kind === "habit"
        ? clampScore(0.55 * activation.score + 0.45 * candidate.strength)
        : candidate.kind === "fact"
          ? clampScore(0.75 * activation.score + 0.25 * candidate.confidence)
          : activation.score;
    // Forgetting curve: availability scales how strongly the memory
    // competes, without changing what it matches.
    const availability = calculateMemoryAvailability({ candidate, now });
    const rankBeforeSpreading = clampScore(baseRank * availability);
    return {
      activationScore: activation.score,
      availability,
      baseRank,
      candidate,
      components: activation.components,
      excludedReason: null,
      query,
      rank: rankBeforeSpreading,
      rankBeforeSpreading,
      spreadingBoost: 0,
    };
  });

  const ranked = scored.filter((entry) => entry.excludedReason === null);

  applySpreadingActivation(ranked);

  const takeKind = (kind: PersonaMemoryCandidate["kind"]) =>
    ranked
      .filter((entry) => entry.candidate.kind === kind)
      .sort((left, right) => {
        if (right.rank !== left.rank) {
          return right.rank - left.rank;
        }
        return (
          right.candidate.createdAt.getTime() -
          left.candidate.createdAt.getTime()
        );
      })
      .slice(0, MEMORY_KIND_BUDGETS[kind])
      .map((entry) => entry.candidate);

  const selected = [
    ...takeKind("fact"),
    ...takeKind("belief"),
    ...takeKind("habit"),
    ...takeKind("style"),
    ...takeKind("episode"),
    ...takeKind("source"),
  ];
  const selectedOrderByKey = new Map(
    selected.map((candidate, index) => [memoryScoreKey(candidate), index + 1])
  );

  return {
    retrieval: {
      budgets: MEMORY_KIND_BUDGETS,
      candidateCount: scored.length,
      candidates: scored.map((entry) =>
        serializeMemoryRetrievalCandidate({
          entry,
          selectedOrder:
            selectedOrderByKey.get(memoryScoreKey(entry.candidate)) ?? null,
        })
      ),
      generatedAt: now.toISOString(),
      queries: input.context.retrievalQueries,
      selectedCount: selected.length,
      version: 1,
    },
    selected,
  };
}
