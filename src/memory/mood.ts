import { and, eq } from "drizzle-orm";

import type { PersonaDatabase as Database } from "../db";
import { personaMoodStates } from "../schema";
import { MAX_EVIDENCE_SUMMARY_CHARS } from "./types";
import type {
  PersonaAffectVector,
  PersonaMemoryCandidate,
  PersonaMoodEmotionLabel,
  PersonaMoodSignal,
  PersonaMoodUpdateTrace,
  PersonaQueryType,
  PersonaTurnAffectEstimate,
  PersonaTurnPlan,
} from "./types";

export function clampScore(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, Number(value.toFixed(3))));
}

// Persona-neutral resting mood the state drifts back to between sessions.
export const PERSONA_MOOD_BASELINE: PersonaAffectVector = {
  arousal: 0.3,
  dominance: 0.5,
  valence: 0.1,
};
// How much of the prior mood survives one turn's emotional impulse.
const MOOD_INERTIA = 0.75;
// Mood drifts halfway back to baseline in a day of silence.
const MOOD_DECAY_HALF_LIFE_HOURS = 24;
const MS_PER_HOUR = 3_600_000;
const MAX_MOOD_LABELS = 8;

function clampAffect(value: PersonaAffectVector): PersonaAffectVector {
  const clamp = (input: number, min: number, max: number) =>
    Math.max(min, Math.min(max, Number.isFinite(input) ? input : min));
  return {
    arousal: clamp(value.arousal, 0, 1),
    dominance: clamp(value.dominance, 0, 1),
    valence: clamp(value.valence, -1, 1),
  };
}

function moodDecay(input: {
  current: PersonaAffectVector | null;
  elapsedMs: number;
}): { decayFactor: number; decayedMood: PersonaAffectVector } {
  const current = input.current ?? PERSONA_MOOD_BASELINE;
  const decayFactor =
    0.5 **
    (Math.max(0, input.elapsedMs) / (MOOD_DECAY_HALF_LIFE_HOURS * MS_PER_HOUR));
  return {
    decayFactor,
    decayedMood: clampAffect({
      arousal:
        PERSONA_MOOD_BASELINE.arousal +
        (current.arousal - PERSONA_MOOD_BASELINE.arousal) * decayFactor,
      dominance:
        PERSONA_MOOD_BASELINE.dominance +
        (current.dominance - PERSONA_MOOD_BASELINE.dominance) * decayFactor,
      valence:
        PERSONA_MOOD_BASELINE.valence +
        (current.valence - PERSONA_MOOD_BASELINE.valence) * decayFactor,
    }),
  };
}

function queryTypeAffectCue(
  queryType: PersonaQueryType
): { affect: PersonaAffectVector; label: string; reason: string } | null {
  if (queryType === "emotional_reflection") {
    return {
      affect: { arousal: 0.7, dominance: 0.45, valence: -0.1 },
      label: "emotional_intensity",
      reason: "The turn was classified as emotional_reflection.",
    };
  }
  if (queryType === "adversarial") {
    return {
      affect: { arousal: 0.65, dominance: 0.6, valence: -0.35 },
      label: "defensiveness",
      reason: "The turn was classified as adversarial.",
    };
  }
  return null;
}

function normalizedEmotionLabel(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, "_");
}

export function emotionLabelFamily(value: string): string {
  const label = normalizedEmotionLabel(value);
  if (
    [
      "anxiety",
      "anxious",
      "fear",
      "fearful",
      "worry",
      "worried",
      "불안",
      "두려움",
      "걱정",
    ].includes(label)
  ) {
    return "anxiety";
  }
  if (
    [
      "sadness",
      "sad",
      "grief",
      "loss",
      "regret",
      "슬픔",
      "상실",
      "후회",
      "그리움",
    ].includes(label)
  ) {
    return "sadness";
  }
  if (
    [
      "anger",
      "angry",
      "frustration",
      "frustrated",
      "분노",
      "화",
      "좌절",
    ].includes(label)
  ) {
    return "anger";
  }
  if (
    [
      "joy",
      "happy",
      "happiness",
      "excitement",
      "excited",
      "delight",
      "기쁨",
      "행복",
      "설렘",
    ].includes(label)
  ) {
    return "joy";
  }
  if (
    ["confidence", "confident", "pride", "자신감", "자부심"].includes(label)
  ) {
    return "confidence";
  }
  if (
    ["calm", "relief", "trust", "차분", "평온", "안도", "신뢰"].includes(label)
  ) {
    return "calm";
  }
  if (["curiosity", "curious", "interest", "호기심", "관심"].includes(label)) {
    return "curiosity";
  }
  return label;
}

function mergeEmotionLabels(
  labels: PersonaMoodEmotionLabel[]
): PersonaMoodEmotionLabel[] {
  const byLabel = new Map<string, PersonaMoodEmotionLabel>();
  for (const label of labels) {
    const normalized = normalizedEmotionLabel(label.label);
    if (!normalized) {
      continue;
    }
    const next = {
      ...label,
      intensity: clampScore(label.intensity),
      label: normalized,
    };
    const current = byLabel.get(normalized);
    if (!current || next.intensity > current.intensity) {
      byLabel.set(normalized, next);
    }
  }
  return [...byLabel.values()]
    .sort((left, right) => {
      if (right.intensity !== left.intensity) {
        return right.intensity - left.intensity;
      }
      return left.label.localeCompare(right.label);
    })
    .slice(0, MAX_MOOD_LABELS);
}

function labelsFromEmotionAnnotations(input: {
  candidate: PersonaMemoryCandidate;
}): PersonaMoodEmotionLabel[] {
  return input.candidate.emotions.map((emotion) => ({
    ...(emotion.evidence ? { evidence: emotion.evidence } : {}),
    intensity: clampScore(emotion.intensity * input.candidate.sourceConfidence),
    label: emotion.emotion,
    memoryId: input.candidate.id,
    source: "activated_memory",
  }));
}

function affectCueFromEmotionFamily(
  family: string
): PersonaAffectVector | null {
  if (family === "anxiety") {
    return { arousal: 0.75, dominance: 0.3, valence: -0.45 };
  }
  if (family === "sadness") {
    return { arousal: 0.35, dominance: 0.3, valence: -0.55 };
  }
  if (family === "anger") {
    return { arousal: 0.75, dominance: 0.65, valence: -0.5 };
  }
  if (family === "joy") {
    return { arousal: 0.65, dominance: 0.55, valence: 0.55 };
  }
  if (family === "confidence") {
    return { arousal: 0.45, dominance: 0.75, valence: 0.4 };
  }
  if (family === "calm") {
    return { arousal: 0.2, dominance: 0.55, valence: 0.25 };
  }
  if (family === "curiosity") {
    return { arousal: 0.55, dominance: 0.55, valence: 0.25 };
  }
  return null;
}

function averageWeightedAffects(
  entries: Array<{ affect: PersonaAffectVector; weight: number }>
): PersonaAffectVector | null {
  const totalWeight = entries.reduce((sum, entry) => sum + entry.weight, 0);
  if (totalWeight <= 0) {
    return null;
  }
  return clampAffect({
    arousal:
      entries.reduce(
        (sum, entry) => sum + entry.affect.arousal * entry.weight,
        0
      ) / totalWeight,
    dominance:
      entries.reduce(
        (sum, entry) => sum + entry.affect.dominance * entry.weight,
        0
      ) / totalWeight,
    valence:
      entries.reduce(
        (sum, entry) => sum + entry.affect.valence * entry.weight,
        0
      ) / totalWeight,
  });
}

function emotionLabelAffectSignals(
  labels: PersonaMoodEmotionLabel[]
): Array<{ affect: PersonaAffectVector; label: PersonaMoodEmotionLabel }> {
  return labels.flatMap((label) => {
    const affect = affectCueFromEmotionFamily(emotionLabelFamily(label.label));
    return affect ? [{ affect, label }] : [];
  });
}

function inferPadEmotionLabel(
  mood: PersonaAffectVector
): PersonaMoodEmotionLabel {
  const label =
    mood.valence <= -0.25 && mood.arousal >= 0.6 && mood.dominance >= 0.55
      ? "anger"
      : mood.valence <= -0.25 && mood.arousal >= 0.6
        ? "anxiety"
        : mood.valence <= -0.25 && mood.arousal < 0.45
          ? "sadness"
          : mood.valence >= 0.25 && mood.arousal >= 0.6
            ? "excitement"
            : mood.valence >= 0.25 && mood.dominance >= 0.6
              ? "confidence"
              : mood.arousal <= 0.35 && mood.valence >= 0
                ? "calm"
                : "neutral";
  const intensity = clampScore(
    Math.max(
      Math.abs(mood.valence),
      Math.abs(mood.arousal - PERSONA_MOOD_BASELINE.arousal),
      Math.abs(mood.dominance - PERSONA_MOOD_BASELINE.dominance)
    )
  );
  return {
    evidence: "Inferred from the resulting PAD mood vector.",
    intensity: Math.max(0.25, intensity),
    label,
    source: "pad_inference",
  };
}

// Estimate this turn's emotional impulse from the activated episodes'
// salience annotations plus coarse query-type cues. Returns null when the
// turn carries no usable affect signal.
export function estimateTurnAffect(input: {
  selected: PersonaMemoryCandidate[];
  turnPlan: PersonaTurnPlan;
}): PersonaAffectVector | null {
  return explainTurnAffect(input).impulse;
}

export function explainTurnAffect(input: {
  selected: PersonaMemoryCandidate[];
  turnPlan: PersonaTurnPlan;
}): PersonaTurnAffectEstimate {
  const annotated = input.selected.filter(
    (
      candidate
    ): candidate is PersonaMemoryCandidate & {
      affect: PersonaAffectVector;
    } => candidate.affect !== null
  );
  const queryType = input.turnPlan.gate.queryType;
  const queryCue = queryTypeAffectCue(queryType);
  const userEmotionLabels: PersonaMoodEmotionLabel[] = [
    ...input.turnPlan.context.contextKeys.emotions.map((emotion) => ({
      intensity: 0.6,
      label: emotion,
      source: "context_emotion" as const,
    })),
    ...input.turnPlan.gate.queryFocus.emotionHints.map((emotion) => ({
      intensity: 0.55,
      label: emotion,
      source: "query_emotion_hint" as const,
    })),
  ];
  const labelSignals = emotionLabelAffectSignals(userEmotionLabels);
  const labelImpulse = averageWeightedAffects(
    labelSignals.map((signal) => ({
      affect: signal.affect,
      weight: Math.max(0.1, signal.label.intensity),
    }))
  );
  const labels: PersonaMoodEmotionLabel[] = [
    ...userEmotionLabels,
    ...input.selected.flatMap((candidate) =>
      labelsFromEmotionAnnotations({ candidate })
    ),
    ...(queryCue
      ? [
          {
            evidence: queryCue.reason,
            intensity: Math.max(
              Math.abs(queryCue.affect.valence),
              queryCue.affect.arousal
            ),
            label: queryCue.label,
            source: "query_type" as const,
          },
        ]
      : []),
  ];
  const reasons: string[] = [];
  const sources: PersonaMoodSignal[] = [
    ...annotated.map((candidate) => ({
      affect: candidate.affect,
      emotionLabels: labelsFromEmotionAnnotations({ candidate }),
      id: candidate.id,
      kind: "activated_memory" as const,
      summary: truncateLine(candidate.summary, MAX_EVIDENCE_SUMMARY_CHARS),
    })),
    ...labelSignals.map((signal) => ({
      affect: signal.affect,
      emotionLabels: [signal.label],
      kind:
        signal.label.source === "query_emotion_hint"
          ? ("query_emotion_hint" as const)
          : ("context_emotion" as const),
      label: signal.label.label,
      summary: `The turn carried ${signal.label.source}=${signal.label.label}.`,
    })),
    ...(queryCue
      ? [
          {
            affect: queryCue.affect,
            emotionLabels: [
              {
                evidence: queryCue.reason,
                intensity: Math.max(
                  Math.abs(queryCue.affect.valence),
                  queryCue.affect.arousal
                ),
                label: queryCue.label,
                source: "query_type" as const,
              },
            ],
            kind: "query_type" as const,
            queryType,
            summary: queryCue.reason,
          },
        ]
      : []),
  ];

  const memoryAffect =
    annotated.length === 0
      ? null
      : clampAffect({
          arousal:
            annotated.reduce(
              (sum, candidate) => sum + candidate.affect.arousal,
              0
            ) / annotated.length,
          dominance:
            annotated.reduce(
              (sum, candidate) => sum + candidate.affect.dominance,
              0
            ) / annotated.length,
          valence:
            annotated.reduce(
              (sum, candidate) => sum + candidate.affect.valence,
              0
            ) / annotated.length,
        });
  const impulse = averageWeightedAffects([
    ...(memoryAffect ? [{ affect: memoryAffect, weight: 1 }] : []),
    ...(queryCue ? [{ affect: queryCue.affect, weight: 1 }] : []),
    ...(labelImpulse ? [{ affect: labelImpulse, weight: 0.75 }] : []),
  ]);

  if (annotated.length > 0) {
    reasons.push(
      `Averaged ${annotated.length} activated memory affect annotation${
        annotated.length === 1 ? "" : "s"
      }.`
    );
  }
  if (queryCue) {
    reasons.push(queryCue.reason);
  }
  if (labelSignals.length > 0) {
    reasons.push(
      `Mapped ${labelSignals.length} context emotion label${
        labelSignals.length === 1 ? "" : "s"
      } into an immediate PAD cue.`
    );
  } else if (userEmotionLabels.length > 0) {
    reasons.push("Recorded emotion labels did not map to a PAD cue.");
  }
  if (annotated.length > 0 && (queryCue || labelImpulse)) {
    reasons.push("Blended memory affect with current-turn emotional cues.");
  }
  if (!impulse) {
    reasons.push(
      "No activated memory affect or current-turn cue changed mood."
    );
  }

  return {
    emotionLabels: mergeEmotionLabels(labels),
    impulse,
    reasons,
    sources,
  };
}

// Mood update reducer: decay the stored mood toward baseline for the elapsed
// silence, then blend in this turn's impulse with inertia. Pure; persistence
// happens in the runtime path.
export function updatePersonaMood(input: {
  current: PersonaAffectVector | null;
  elapsedMs: number;
  impulse: PersonaAffectVector | null;
}): PersonaAffectVector {
  return calculatePersonaMoodUpdate(input).mood;
}

export function calculatePersonaMoodUpdate(input: {
  current: PersonaAffectVector | null;
  elapsedMs: number;
  impulse: PersonaAffectVector | null;
  turnAffect?: PersonaTurnAffectEstimate | null;
}): { mood: PersonaAffectVector; trace: PersonaMoodUpdateTrace } {
  const { decayFactor, decayedMood } = moodDecay(input);
  const mood = input.impulse
    ? clampAffect({
        arousal:
          decayedMood.arousal * MOOD_INERTIA +
          input.impulse.arousal * (1 - MOOD_INERTIA),
        dominance:
          decayedMood.dominance * MOOD_INERTIA +
          input.impulse.dominance * (1 - MOOD_INERTIA),
        valence:
          decayedMood.valence * MOOD_INERTIA +
          input.impulse.valence * (1 - MOOD_INERTIA),
      })
    : decayedMood;
  const reasons = [
    input.current
      ? `Decayed the previous mood toward baseline over ${Math.max(
          0,
          input.elapsedMs
        )}ms.`
      : "Started from the persona mood baseline because no prior mood was stored.",
    ...(input.turnAffect?.reasons ?? []),
    input.impulse
      ? `Blended decayed mood with the turn impulse using ${MOOD_INERTIA.toFixed(
          2
        )} inertia.`
      : "No turn impulse was applied after decay.",
  ];
  const emotionLabels = mergeEmotionLabels([
    ...(input.turnAffect?.emotionLabels ?? []),
    inferPadEmotionLabel(mood),
  ]);
  return {
    mood,
    trace: {
      baseline: { ...PERSONA_MOOD_BASELINE },
      decayFactor,
      decayHalfLifeHours: MOOD_DECAY_HALF_LIFE_HOURS,
      decayedMood,
      elapsedMs: Math.max(0, input.elapsedMs),
      emotionLabels,
      impulse: input.impulse,
      inertia: MOOD_INERTIA,
      previousMood: input.current,
      reasons,
      result: mood,
      sources: input.turnAffect?.sources ?? [],
      version: 1,
    },
  };
}
export function truncateLine(line: string, maxChars: number): string {
  if (line.length <= maxChars) {
    return line;
  }
  let sliced = line.slice(0, maxChars - 2);
  // Don't split a surrogate pair; a lone surrogate is invalid once the prompt
  // is serialized for the model API.
  const lastCode = sliced.charCodeAt(sliced.length - 1);
  if (lastCode >= 0xd8_00 && lastCode <= 0xdb_ff) {
    sliced = sliced.slice(0, -1);
  }
  return `${sliced} …`;
}

export async function loadPersonaMoodState(
  db: Database,
  input: { tenantId: string; personaId: string; userId: string }
) {
  const [row] = await db
    .select()
    .from(personaMoodStates)
    .where(
      and(
        eq(personaMoodStates.tenantId, input.tenantId),
        eq(personaMoodStates.personaId, input.personaId),
        eq(personaMoodStates.userId, input.userId)
      )
    )
    .limit(1);
  return row ?? null;
}

export async function persistPersonaMoodState(
  db: Database,
  input: {
    mood: PersonaAffectVector;
    tenantId: string;
    personaId: string;
    turnCount: number;
    userId: string;
  }
): Promise<void> {
  await db
    .insert(personaMoodStates)
    .values({
      arousal: input.mood.arousal,
      dominance: input.mood.dominance,
      tenantId: input.tenantId,
      personaId: input.personaId,
      turnCount: input.turnCount,
      userId: input.userId,
      valence: input.mood.valence,
    })
    .onConflictDoUpdate({
      set: {
        arousal: input.mood.arousal,
        dominance: input.mood.dominance,
        turnCount: input.turnCount,
        updatedAt: new Date(),
        valence: input.mood.valence,
      },
      target: [
        personaMoodStates.tenantId,
        personaMoodStates.personaId,
        personaMoodStates.userId,
      ],
    });
}
