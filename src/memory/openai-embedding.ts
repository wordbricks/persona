import { z } from "zod";

export const PERSONA_EMBEDDING_MODEL = "text-embedding-3-small";

const EmbeddingResponseSchema = z.object({
  data: z.array(
    z.object({
      embedding: z.array(z.number()).optional(),
      index: z.number().optional(),
    })
  ),
  usage: z
    .object({
      prompt_tokens: z.number(),
      total_tokens: z.number(),
    })
    .optional(),
});

export async function generateEmbeddings(
  texts: string[],
  openaiApiKey: string
): Promise<number[][]> {
  if (texts.length === 0) {
    return [];
  }

  const response = await fetch("https://api.openai.com/v1/embeddings", {
    body: JSON.stringify({
      model: PERSONA_EMBEDDING_MODEL,
      input: texts,
    }),
    headers: {
      Authorization: `Bearer ${openaiApiKey}`,
      "Content-Type": "application/json",
    },
    method: "POST",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`
    );
  }

  const json = await response.json();
  const parseResult = EmbeddingResponseSchema.safeParse(json);
  if (!parseResult.success) {
    throw new Error("Invalid response from OpenAI embeddings API");
  }

  const ordered = [...parseResult.data.data].sort(
    (left, right) => (left.index ?? 0) - (right.index ?? 0)
  );
  const embeddings = ordered.map((entry) => entry.embedding);
  if (
    embeddings.length !== texts.length ||
    embeddings.some((embedding) => !embedding)
  ) {
    throw new Error("OpenAI embeddings API returned an incomplete batch");
  }

  return embeddings as number[][];
}

export async function generateEmbedding(
  text: string,
  openaiApiKey: string
): Promise<number[]> {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    body: JSON.stringify({
      model: PERSONA_EMBEDDING_MODEL,
      input: text,
    }),
    headers: {
      Authorization: `Bearer ${openaiApiKey}`,
      "Content-Type": "application/json",
    },
    method: "POST",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`
    );
  }

  const json = await response.json();
  const parseResult = EmbeddingResponseSchema.safeParse(json);
  if (!parseResult.success) {
    throw new Error("Invalid response from OpenAI embeddings API");
  }

  const embedding = parseResult.data.data?.[0]?.embedding;
  if (!embedding) {
    throw new Error("No embedding returned from OpenAI");
  }

  return embedding;
}
