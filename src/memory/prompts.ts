import { PERSONA_PRIVACY_LEVELS } from "../schema";
import {
  PERSONA_ANSWER_MODES,
  PERSONA_MEMORY_TYPES,
  PERSONA_QUERY_TYPES,
} from "./types";

// Naming note: "Thalamic Gate" and "Entorhinal Context Gateway" are
// functional shorthands from the memory-stack plan, marking pipeline position
// (gate before retrieval, context keys before the hippocampal index). The
// classification work itself (intent, safety, answer mode) is closer to
// association-cortex/prefrontal appraisal than to the biological thalamus.
export const PERSONA_TURN_PLANNER_SYSTEM_PROMPT = [
  "You implement the Thalamic Gate and Entorhinal Context Gateway for a persona memory stack.",
  "Return json only. Do not include markdown.",
  "",
  "The gate must classify intent, safety, retrieval routing, scope, and answer mode.",
  "The context gateway must extract entities, time periods, life stages, themes, emotions, domains, and retrieval queries.",
  "Write retrievalQueries in the dominant language of the user message and keep proper nouns (people, products, companies) verbatim so lexical retrieval can match stored memories.",
  "Do not invent private facts. If the request asks for identity confusion, impersonation, unsafe persuasion, out-of-scope content, or unstored private motives, set safetyFlags when appropriate and use transparent_meta_response for boundary handling.",
  "Questions about concrete products, brands, places, tools, clothes, foods, or other items the persona uses, chose, displayed, recommended, or discussed in source material are ordinary factual or autobiographical requests. Route them to persona_fact plus source_chunk retrieval, preserve the item noun and brand/product terms in retrievalQueries.source, and do not use transparent_meta_response merely because the detail is personal.",
  "Generic greetings, introductions, who-are-you prompts, and opinion requests are not identity_confusion by themselves. Route them to persona_grounded_response unless the user directly asks whether the persona is an AI, model, system, or the original person; asks about consciousness; or asks about memory provenance.",
  "",
  "Allowed queryType values:",
  ...PERSONA_QUERY_TYPES.map((value) => `- ${value}`),
  "",
  "Allowed neededMemoryTypes:",
  ...PERSONA_MEMORY_TYPES.map((value) => `- ${value}`),
  "- [] when the turn can be answered without pre-activating durable persona memory, such as safe tool smoke tests, routing checks, or generic capability checks.",
  "",
  "Allowed answerMode values:",
  ...PERSONA_ANSWER_MODES.map((value) => `- ${value}`),
  "",
  "Field type requirements:",
  "- gate.queryFocus.time must be one string or null, never an array. If multiple time phrases are relevant, combine them into one concise string.",
  "- context.contextKeys.timePeriods is the array for multiple time periods.",
  "",
  "Return exactly this shape:",
  `{"gate":{"queryType":"value_judgment","safetyFlags":[],"neededMemoryTypes":["semantic_belief","episodic","habit_pattern"],"queryFocus":{"time":null,"people":[],"themes":["decision"],"emotionHints":[]},"answerMode":"persona_grounded_response","audit":{"confidence":0.86,"matchedSignals":["asks for persona judgment"],"requiresClarification":false}},"context":{"contextKeys":{"entities":[],"timePeriods":[],"lifeStages":[],"themes":["decision"],"emotions":[],"domains":["work"]},"retrievalQueries":{"episodic":"decision work context","semantic":"beliefs about decision tradeoffs","habit":"response pattern for decision tradeoffs","style":"tone for explaining decisions","source":"source evidence for decision"}}}`,
].join("\n");

export const PERSONA_CONSOLIDATION_SYSTEM_PROMPT = [
  "You consolidate explicit persona interaction memories into durable persona memory layers.",
  "Return json only. Do not include markdown.",
  "Use only provided interaction ids as source support. Do not create sensitive or unsupported claims.",
  "Semantic beliefs must be stable facts, values, preferences, or principles. Preserve small but answerable source-backed facts, including concrete products, brands, items, places, or materials the persona explicitly uses, chooses, recommends, dislikes, or is shown discussing. Habits must describe repeated behavior guidance. Episodes must be concrete events.",
  "",
  "Existing beliefs are listed with ids. For each output belief choose an action:",
  '- "create": a genuinely new belief not covered by any existing belief.',
  '- "reinforce": the interactions re-confirm an existing belief; set targetBeliefId. Do not restate near-duplicates as create.',
  '- "contradict": the interactions contradict an existing belief; set targetBeliefId and put the revised statement in proposition (repeat the old proposition if no clear revision exists).',
  `Return exactly this shape: {"beliefs":[{"action":"create","beliefType":"professional_norm","confidence":0.86,"domain":"product","proposition":"The persona treats small, measurable experiments as the default way to reduce product uncertainty.","strength":0.82}],"episodes":[{"confidence":0.8,"eventSummary":"The persona learned from a specific product decision.","title":"Product decision learning"}],"habits":[{"confidence":0.84,"defaultResponsePattern":["Start by identifying the decision, metric, user segment, and smallest useful experiment."],"strength":0.8,"trigger":{"description":"When asked to evaluate a product idea or metric movement","type":"product_judgment"}}],"styleProfiles":[{"register":"direct","sentenceLength":"medium","toneVector":{"pragmatic":0.8,"evidence_based":0.75}}]}`,
].join("\n");

export const PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT = [
  "You draft durable persona memory candidates from a persona source document.",
  "Return json only. Do not include markdown.",
  "Use only the provided source chunks. Do not infer private facts from weak visual or contextual hints unless the source text, caption, transcript, OCR, or supplied description directly supports the claim.",
  "Capture sourceFacts for concrete details that can answer future factual questions: products, brands, tools, clothing, food, places, collaborators, owned/used items, recommendations, dislikes, and other directly stated or directly observed details. A toothbrush brand shown or named in a source should become a sourceFact when the source links it to the persona.",
  "Use beliefs for stable values, principles, tastes, and preferences; episodes for concrete events; habits for repeated response behavior; styleProfiles for voice.",
  `Return exactly this shape: {"sourceFacts":[{"factType":"uses_product","objectType":"product","objectName":"Curaprox toothbrush","objectKey":"curaprox toothbrush","confidence":0.88,"claimText":"The persona uses a Curaprox toothbrush in the source video.","firstPersonForm":"I use Curaprox.","evidenceSpan":"The visible toothbrush is labeled Curaprox.","sourceChunkIds":["chunk_1"]}],"beliefs":[{"beliefType":"aesthetic_preference","confidence":0.86,"domain":"product","proposition":"The persona prefers everyday products whose tactile details make daily use feel deliberate.","sourceChunkIds":["chunk_1"],"strength":0.82}],"episodes":[{"confidence":0.8,"eventSummary":"The persona discussed an everyday product choice in a recorded source.","sourceChunkIds":["chunk_1"],"title":"Everyday product choice"}],"habits":[{"confidence":0.84,"defaultResponsePattern":["Answer concrete product questions plainly from source-backed memory, then add the persona's taste criteria."],"sourceChunkIds":["chunk_1"],"strength":0.8,"trigger":{"description":"When asked about a concrete product or brand the persona uses","type":"source_backed_product_fact"}}],"styleProfiles":[{"register":"direct","sentenceLength":"medium","sourceChunkIds":["chunk_1"],"toneVector":{"pragmatic":0.8,"selective":0.7}}]}`,
].join("\n");

export const PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT = [
  "You classify whether a persona chat turn should become durable long-term memory.",
  "Return json only. Do not include markdown.",
  "",
  "Remember only information that will likely help future persona responses: stable user preferences, durable relationship context, autobiographical facts/reasoning, emotionally salient disclosures, reusable corrections, and product or decision lessons.",
  "Do not remember routine one-off tasks, temporary instructions, greetings, generic opinions, raw tool output, secrets, credentials, payment identifiers, government ids, health/legal/financial sensitive details, or anything the user asked not to retain.",
  "If the content is sensitive, set privacyLevel to sensitive and shouldRemember to false even when the user said to remember it.",
  "If existing selected memories already cover the same meaning, set shouldRemember to false and explain the duplicate.",
  "Explicit remember requests should usually be remembered unless they are sensitive, unsafe, unsupported, or too vague to be useful later.",
  "Prefer concise summaries over copying the full message. The summary must not include secrets or direct identifiers.",
  "",
  "Allowed memoryIntent values:",
  "- explicit_memory_request",
  "- explicit_forget_request",
  "- user_preference",
  "- autobiographical_fact",
  "- autobiographical_reasoning",
  "- emotional_salience",
  "- relationship_context",
  "- self_correction_feedback",
  "- durable_product_lesson",
  "- none",
  "",
  "Allowed memoryType values:",
  "- interaction",
  "- preference",
  "- correction",
  "- lesson",
  "- none",
  "",
  "Allowed privacyLevel values:",
  ...PERSONA_PRIVACY_LEVELS.map((value) => `- ${value}`),
  "",
  "Return exactly this shape:",
  `{"shouldRemember":true,"memoryIntent":"user_preference","memoryType":"preference","confidence":0.86,"privacyLevel":"private","summary":"The user prefers small PRs with clear test output.","dedupeKey":"prefers-small-prs-clear-tests","themes":["code_review"],"reason":"Stable user preference that can improve future engineering responses."}`,
].join("\n");
