# @wordbricks/persona

![npm version](https://img.shields.io/npm/v/@wordbricks/persona)
![Apache-2.0 license](https://img.shields.io/badge/license-Apache--2.0-blue)

`@wordbricks/persona` is a persona memory and runtime library for LLM agents. It gives an application a durable memory substrate, a turn planner, prompt context assembly, and post-response memory review without owning your database connection, model provider, embedder, chat stack, or deployment runtime.

The memory model is brain-inspired rather than a single global vector search. Tonic identity memory - beliefs, habits, and style - has independent selection budgets from phasic episodic/source memory, so stable identity does not get crowded out by recent events. Recall combines lexical overlap with pgvector cosine similarity, then applies one-hop spreading activation across linked memories. Episodic availability follows an exponential forgetting curve, but recent activation and emotional salience can strengthen recall. Mood is represented as PAD (`valence`, `arousal`, `dominance`) and decays between turns. Beliefs are reconsolidated through reinforcement and contradiction, not overwritten. During generation, an agent can perform re-entrant recall by calling back into memory with a specific cue.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the retrieval, forgetting, mood, and reconsolidation details.

## Design

This package is bring-your-own infrastructure:

- Bring your own Postgres database with pgvector. The schema is implemented with Drizzle and exported from `@wordbricks/persona/schema`.
- Bring your own Drizzle database handle. `drizzle-orm` is a peer dependency so your app and this package share one Drizzle instance.
- Bring your own LLM. Planning, triage, and consolidation use the `PersonaJsonLlm` callback: `{ systemPrompt, userPrompt } => Promise<unknown>`.
- Bring your own embedder. Retrieval works lexically without embeddings, or semantically with any `PersonaEmbedder`; the package includes `createOpenAiPersonaEmbedder` for OpenAI embeddings.
- Bring your own chat runtime. `buildPersonaInstructions` returns instructions that you pass to your normal agent or chat-completion stack.
- Optionally attach an external memory service. The Hindsight adapter can recall, retain, and reflect through `createHindsightPersonaMemoryClient`.
- Use `PersonaLogger` and `defer` hooks for serverless runtimes. In Cloudflare Workers, pass `defer: ctx.waitUntil` so background retain/review work can finish after the response.

## Requirements

- Node.js or Bun
- Postgres with pgvector enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Install the package and the shared Drizzle peer:

```sh
npm i @wordbricks/persona drizzle-orm
```

You will also need your normal Drizzle database driver and migration tooling, for example `postgres` and `drizzle-kit`.

## Schema Setup

Re-export the persona schema from your app and include that file in your `drizzle-kit` config. This lets your app own migrations while keeping table definitions sourced from the package.

```ts
// src/db/persona-schema.ts
export * from "@wordbricks/persona/schema";
```

```ts
// drizzle.config.ts
import { defineConfig } from "drizzle-kit";

export default defineConfig({
  dialect: "postgresql",
  out: "./drizzle",
  schema: ["./src/db/schema.ts", "./src/db/persona-schema.ts"],
});
```

Then generate and apply migrations through your normal Drizzle workflow.

`organizationId`, `userId`, `chatSessionId`, and `chatMessageId` are opaque string scoping columns. The package intentionally does not declare foreign keys to host-app tables. If your app wants referential constraints, add them in your own migrations.

Embeddings are stored at `PERSONA_EMBEDDING_DIMENSION` (`1536`), matching OpenAI `text-embedding-3-small` through the built-in helper. If you use a different embedder, keep the schema dimension aligned with that embedder.

## Quickstart

This is the full wiring shape. The same flow is typechecked in [examples/basic.ts](./examples/basic.ts).

```ts
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import {
  buildPersonaInstructions,
  createOpenAiPersonaEmbedder,
  ingestPersonaSourceDocument,
  preparePersonaRuntimeContext,
  processPersonaMemoryConsolidationTick,
  recordPostResponsePersonaMemoryReview,
  rememberPersonaLayerMemory,
  upsertPersonaProfile,
} from "@wordbricks/persona";
import type { PersonaDatabase, PersonaJsonLlm } from "@wordbricks/persona";
import * as personaSchema from "@wordbricks/persona/schema";

const sql = postgres(process.env.DATABASE_URL!);
const db = drizzle(sql, { schema: personaSchema }) as PersonaDatabase;

const organizationId = "org_123";
const personaKey = "product-coach";
const userId = "user_123";
const openAiApiKey = process.env.OPENAI_API_KEY!;
const embed = createOpenAiPersonaEmbedder(openAiApiKey);

const personaJsonLlm: PersonaJsonLlm = async ({ systemPrompt, userPrompt }) => {
  const response = await fetch("https://api.example.com/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.JSON_LLM_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "your-json-mode-model",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: `${userPrompt}\n\nReturn JSON only.` },
      ],
      temperature: 0,
    }),
  });
  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | null } }>;
  };
  const content = payload.choices?.[0]?.message?.content;
  if (!content) throw new Error("Persona JSON LLM returned no content.");
  return JSON.parse(content) as unknown;
};

const persona = await upsertPersonaProfile(db, {
  organizationId,
  personaKey,
  displayName: "Product Coach",
  personaType: "synthetic_role",
  consentStatus: "fictional_or_authorized",
  policy: {
    allowedUse: ["Help teams reason about product decisions."],
    forbiddenUse: ["Do not present this persona as a real person."],
    transparencyLabel: "AI persona simulation for Product Coach.",
  },
  profile: {
    voice: "plain-spoken, rigorous, and concrete",
  },
  updatedByUserId: userId,
});

await ingestPersonaSourceDocument(db, {
  organizationId,
  personaKey,
  title: "Product Coach Seed Notes",
  rawText:
    "The Product Coach prefers writing down the user problem, the bet, and the fastest falsifying signal before committing engineering time.",
  sourceType: "seed",
  sourcePriority: "synthetic",
  rightsStatus: "owned",
  embed,
});

await rememberPersonaLayerMemory(db, {
  organizationId,
  personaKey,
  userId,
  updatedByUserId: userId,
  memoryKind: "habit",
  title: "Product review habit",
  summary:
    "When reviewing product ideas, the persona asks for the riskiest assumption and the smallest credible test.",
  content: {
    triggerDescription: "When asked to review a product idea",
    defaultResponsePattern: [
      "Name the assumption, name the user evidence, then suggest the smallest test.",
    ],
  },
  embed,
});

const userMessage = "Should we build a dashboard for this feature first?";
const runtime = await preparePersonaRuntimeContext(db, {
  organizationId,
  personaKey,
  userId,
  message: userMessage,
  disclosurePolicy: "always",
  llm: personaJsonLlm,
  embed,
});

const systemPrompt = buildPersonaInstructions({
  personaKey,
  language: "en",
  disclosurePolicy: runtime.disclosurePolicy,
  personaPromptContext: runtime.promptContext,
  turnPlan: runtime.turnPlan,
});

const answer = await yourChatLlm({
  system: systemPrompt,
  user: userMessage,
});

await recordPostResponsePersonaMemoryReview(db, {
  organizationId,
  userId,
  userMessage,
  assistantMessage: answer,
  persona: runtime.persona,
  turnPlan: runtime.turnPlan,
  workspaceId: runtime.workspaceId,
  llm: personaJsonLlm,
});

// Run this from a cron or queue worker.
await processPersonaMemoryConsolidationTick({
  db,
  consolidate: personaJsonLlm,
  embed,
});
```

## API Overview

| Area | Module | Key exports |
| --- | --- | --- |
| Profile | `@wordbricks/persona` | `upsertPersonaProfile`, `loadPersonaProfile`, `publishPersonaProfile`, `copyPersonaProfile`, `deletePersonaProfile`, `upsertPersonaAlias`, `listPersonaAliases` |
| Source ingestion | `@wordbricks/persona` | `ingestPersonaSourceDocument`, `chunkPersonaSourceText`, `draftPersonaMemoriesFromSourceDocument`, `activatePersonaDraftMemory`, `rememberPersonaLayerMemory`, `forgetPersonaLayerMemory` |
| Runtime and turn memory | `@wordbricks/persona` | `preparePersonaRuntimeContext`, `planPersonaTurnWithLlm`, `recallPersonaMemoriesForCue`, `recordPostResponsePersonaMemoryReview`, `triagePostResponseInteractionMemoryWithLlm`, `processPersonaMemoryConsolidationTick` |
| Mood and selection | `@wordbricks/persona` | `calculatePersonaMoodUpdate`, `estimateTurnAffect`, `updatePersonaMood`, `selectPersonaMemories`, `selectPersonaMemoriesWithScores`, `calculateMemoryAvailability` |
| Embeddings | `@wordbricks/persona` | `createOpenAiPersonaEmbedder`, `upsertPersonaMemoryEmbeddings`, `backfillPersonaMemoryEmbeddings`, `hashPersonaMemoryText`, `normalizePersonaEmbeddingText` |
| Hindsight adapter | `@wordbricks/persona` | `createHindsightPersonaMemoryConfig`, `createHindsightPersonaMemoryClient`, `createNoopHindsightPersonaMemoryClient`, `hindsightPersonaBankId`, `hindsightPersonaTags` |
| Agent instructions | `@wordbricks/persona/agent` | `buildPersonaInstructions`, `PersonaLanguage` |
| Schema | `@wordbricks/persona/schema` | Drizzle tables, insert/select types, enums, `PERSONA_EMBEDDING_DIMENSION` |

The root export re-exports `./memory`, `./schema`, and `./agent`, so most applications can import from `@wordbricks/persona` until they want a narrower module boundary.

## Responsible Use

Read [RESPONSIBLE_USE.md](./RESPONSIBLE_USE.md) before enabling personas for real users. Persona simulation of real people requires consent, authorization, or a carefully reviewed public-material-only basis. `PERSONA_PROFILE_TYPES` distinguishes fictional/composite characters, living public figures, deceased public figures, private authorized people, and synthetic roles; `PERSONA_CONSENT_STATUSES` records the consent basis. Real-person personas should not be activated with `unknown` consent outside controlled review, and disclosures should make clear that users are interacting with an AI persona system, not the biological person.

## Development

```sh
bun install
bun run test
bun run typecheck
bun run build
```

## License

Apache-2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
