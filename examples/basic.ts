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
} from "../src";
import type { PersonaDatabase, PersonaJsonLlm } from "../src";
import * as personaSchema from "../src/schema";

type ChatCompletionResponse = {
  choices?: Array<{ message?: { content?: string | null } }>;
};

async function readChatCompletionContent(response: Response): Promise<string> {
  if (!response.ok) {
    throw new Error(
      `Chat completion failed: ${response.status} ${await response.text()}`
    );
  }
  const payload = (await response.json()) as ChatCompletionResponse;
  const content = payload.choices?.at(0)?.message?.content?.trim();
  if (!content) {
    throw new Error("Chat completion returned an empty response.");
  }
  return content;
}

async function callJsonModeChat(input: {
  apiKey: string;
  baseUrl?: string;
  model: string;
  systemPrompt: string;
  userPrompt: string;
}): Promise<unknown> {
  const response = await fetch(
    `${input.baseUrl ?? "https://api.openai.com/v1"}/chat/completions`,
    {
      body: JSON.stringify({
        max_tokens: 1400,
        messages: [
          { content: input.systemPrompt, role: "system" },
          { content: `${input.userPrompt}\n\nReturn JSON only.`, role: "user" },
        ],
        model: input.model,
        response_format: { type: "json_object" },
        temperature: 0,
      }),
      headers: {
        Authorization: `Bearer ${input.apiKey}`,
        "Content-Type": "application/json",
      },
      method: "POST",
    }
  );
  return JSON.parse(await readChatCompletionContent(response)) as unknown;
}

async function callTextChat(input: {
  apiKey: string;
  baseUrl?: string;
  model: string;
  systemPrompt: string;
  userPrompt: string;
}): Promise<string> {
  const response = await fetch(
    `${input.baseUrl ?? "https://api.openai.com/v1"}/chat/completions`,
    {
      body: JSON.stringify({
        max_tokens: 2000,
        messages: [
          { content: input.systemPrompt, role: "system" },
          { content: input.userPrompt, role: "user" },
        ],
        model: input.model,
        temperature: 0.4,
      }),
      headers: {
        Authorization: `Bearer ${input.apiKey}`,
        "Content-Type": "application/json",
      },
      method: "POST",
    }
  );
  return readChatCompletionContent(response);
}

export async function runBasicPersonaExample(input: {
  chatModel: string;
  databaseUrl: string;
  jsonModel: string;
  openAiApiKey: string;
}): Promise<string> {
  const sql = postgres(input.databaseUrl);
  const db = drizzle(sql, { schema: personaSchema }) as PersonaDatabase;
  const tenantId = "tenant_example";
  const personaKey = "product-coach";
  const userId = "user_example";
  const embed = createOpenAiPersonaEmbedder(input.openAiApiKey);
  const llm: PersonaJsonLlm = async ({ systemPrompt, userPrompt }) =>
    callJsonModeChat({
      apiKey: input.openAiApiKey,
      model: input.jsonModel,
      systemPrompt,
      userPrompt,
    });

  try {
    await upsertPersonaProfile(db, {
      consentStatus: "fictional_or_authorized",
      displayName: "Product Coach",
      tenantId,
      personaKey,
      personaType: "synthetic_role",
      policy: {
        allowedUse: ["Help teams reason about product decisions."],
        biographicalSummary:
          "A synthetic product coach that values explicit tradeoffs and direct feedback.",
        forbiddenUse: ["Do not present this persona as a real person."],
        knowledgeCutoffForPersona: "2026-01-01",
        transparencyLabel: "AI persona simulation for Product Coach.",
      },
      profile: {
        voice: "plain-spoken, rigorous, and concrete",
      },
      updatedByUserId: userId,
    });

    await ingestPersonaSourceDocument(db, {
      embed,
      tenantId,
      personaKey,
      rawText:
        "The Product Coach prefers writing down the user problem, the bet, and the fastest falsifying signal before committing engineering time.",
      rightsStatus: "owned",
      sourcePriority: "synthetic",
      sourceType: "seed",
      title: "Product Coach Seed Notes",
    });

    await rememberPersonaLayerMemory(db, {
      content: {
        defaultResponsePattern: [
          "Name the assumption, name the user evidence, then suggest the smallest test.",
        ],
        triggerDescription: "When asked to review a product idea",
      },
      embed,
      memoryKind: "habit",
      tenantId,
      personaKey,
      summary:
        "When reviewing product ideas, the persona asks for the riskiest assumption and the smallest credible test.",
      title: "Product review habit",
      updatedByUserId: userId,
      userId,
    });

    const userMessage = "Should we build a dashboard for this feature first?";
    const runtime = await preparePersonaRuntimeContext(db, {
      disclosurePolicy: "always",
      embed,
      llm,
      message: userMessage,
      tenantId,
      personaKey,
      userId,
    });
    const instructions = buildPersonaInstructions({
      disclosurePolicy: runtime.disclosurePolicy,
      language: "en",
      personaKey,
      personaPromptContext: runtime.promptContext,
      turnPlan: runtime.turnPlan,
    });
    const answer = await callTextChat({
      apiKey: input.openAiApiKey,
      model: input.chatModel,
      systemPrompt: instructions,
      userPrompt: userMessage,
    });

    await recordPostResponsePersonaMemoryReview(db, {
      assistantMessage: answer,
      llm,
      tenantId,
      persona: runtime.persona,
      turnPlan: runtime.turnPlan,
      userId,
      userMessage,
      workspaceId: runtime.workspaceId,
    });

    await processPersonaMemoryConsolidationTick({
      consolidate: llm,
      db,
      embed,
    });

    return answer;
  } finally {
    await sql.end();
  }
}
