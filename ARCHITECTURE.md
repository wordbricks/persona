# Persona Memory Architecture

## Tonic And Phasic Memory

Identity memory is tonic, while event and source memory is phasic. `MEMORY_KIND_BUDGETS` reserves independent selection budgets by kind so `belief`, `habit`, and `style` memories stay available alongside `episode`, `fact`, and `source` memories instead of competing in one global pool. `calculateMemoryAvailability` returns `1` for beliefs, habits, and style, while episodic/source-like memories decay and compete on the current turn's retrieval cues.

## Hybrid Recall

Recall combines lexical and vector signals. `tokenize` preserves lowercased English/Korean tokens and strips common Korean particles, while `embeddingSimilarityExpression` retrieves pgvector cosine similarity from `personaMemoryEmbeddings`; `normalizeCosineSimilarity` rescales embedding similarity so it can be compared with lexical overlap. `calculateMemoryActivationScore` then combines semantic, entity, theme, temporal, affective, identity, confidence, and privacy terms before `retrievePersonaMemoriesWithScores` selects candidates.

## One-Hop Spreading Activation

`applySpreadingActivation` runs a single association pass over `linkIds` and `aliasIds`. If one memory links to another, each can receive up to `SPREADING_ACTIVATION_FACTOR` (`0.3`) times the other's rank as `spreadingBoost`. The boost uses `Math.max`, not summation, so dense memory graphs do not snowball, and the one-hop pass keeps activation local to direct associations.

## Forgetting And Emotional Salience

`calculateMemoryAvailability` implements an exponential forgetting curve with `EPISODIC_HALF_LIFE_DAYS` (`90`), `RECALL_PRIMING_HALF_LIFE_DAYS` (`30`), and `MEMORY_AVAILABILITY_FLOOR` (`0.35`). Source confidence lengthens the creation half-life, recent activation adds priming, and `activationCount` adds use-dependent strengthening. Emotional salience influences ranking through `calculateMemoryActivationScore`: episode candidates use `retrievalBoost`, PAD mood congruence, and emotion-label overlap when computing `affectiveSalience`.

## PAD Mood

Mood is represented as PAD (`valence`, `arousal`, `dominance`). `calculatePersonaMoodUpdate` decays stored mood toward `PERSONA_MOOD_BASELINE` (`valence: 0.1`, `arousal: 0.3`, `dominance: 0.5`) with `MOOD_DECAY_HALF_LIFE_HOURS` (`24`), then blends in the current turn's impulse with `MOOD_INERTIA` (`0.75`). `explainTurnAffect` derives the impulse from query cues and activated memories, while `loadPersonaMoodState` and `persistPersonaMoodState` carry the mood between turns.

## Belief Reconsolidation

Beliefs are updated by reinforcement and contradiction rather than overwritten. `applyBeliefReinforcement` uses `BELIEF_REINFORCEMENT_RATE` (`0.3`) to move confidence and strength asymptotically toward certainty. `applyBeliefContradiction` uses `BELIEF_CONTRADICTION_RATE` (`0.45`) to decay confidence and strength; when strength falls below `BELIEF_CONFLICT_THRESHOLD` (`0.35`), the belief becomes `conflicted`. `consolidatePersonaMemoryScope` applies those updates and can create revised beliefs while preserving source ids.

## Re-Entrant Recall

The persona can re-cue memory during a response through the `recall_persona_memory` tool. The tool calls `recallPersonaMemoriesForCue`, which builds a cue-only `PersonaContextGatewayOutput`, retrieves selected memories with the same retrieval stack, and returns formatted memory sections. This lets the agent fill a memory gap discovered while drafting without re-running the full turn planner.
