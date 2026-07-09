import { DEFAULT_PERSONA_LOGGER, personaLogMessage } from "./logger";
import type {
  PersonaLogger,
  PersonaMemoryType,
  PersonaQueryType,
  PersonaTurnPlan,
} from "./types";

export type HindsightPersonaMemoryEnv = {
  HINDSIGHT_ENABLED?: boolean | string;
  HINDSIGHT_API_KEY?: string;
  HINDSIGHT_BASE_URL?: string;
  HINDSIGHT_TIMEOUT_MS?: number | string;
  HINDSIGHT_RECALL_ENABLED?: boolean | string;
  HINDSIGHT_RECALL_TOP_K?: number | string;
  HINDSIGHT_RETAIN_ENABLED?: boolean | string;
  HINDSIGHT_RETAIN_ASYNC_ENABLED?: boolean | string;
  HINDSIGHT_REFLECT_ENABLED?: boolean | string;
  HINDSIGHT_REFLECT_TIMEOUT_MS?: number | string;
  HINDSIGHT_FAIL_OPEN?: boolean | string;
};

export type HindsightPersonaMemoryConfig = {
  apiKey: string | null;
  baseUrl: string | null;
  enabled: boolean;
  failOpen: boolean;
  recallEnabled: boolean;
  recallTopK: number;
  reflectEnabled: boolean;
  reflectTimeoutMs: number;
  retainAsyncEnabled: boolean;
  retainEnabled: boolean;
  timeoutMs: number;
};

export type HindsightRecallInput = {
  contextKeys: PersonaTurnPlan["context"]["contextKeys"];
  desiredMemoryTypes: PersonaMemoryType[];
  maxResults: number;
  message: string;
  organizationId: string;
  personaId: string;
  personaKey: string;
  retrievalQueries: PersonaTurnPlan["context"]["retrievalQueries"];
  turnQueryType: PersonaQueryType;
  userId: string;
};

export type HindsightRawRecallMemory = {
  bankId: string;
  chunkId: string | null;
  context: string | null;
  documentId: string | null;
  entities: string[];
  id: string;
  mentionedAt: string | null;
  metadata: Record<string, string>;
  occurredEnd: string | null;
  occurredStart: string | null;
  sourceFactIds: string[];
  tags: string[];
  text: string;
  type: string | null;
};

export type HindsightRecallResult = {
  attempted: boolean;
  enabled: boolean;
  latencyMs?: number;
  memories: HindsightRawRecallMemory[];
  provider: "hindsight";
  skippedReason?: string;
};

export type HindsightRetainInput = {
  chatMessageId?: string | null;
  chatSessionId?: string | null;
  content: string;
  context?: string | null;
  documentId?: string | null;
  organizationId: string;
  personaId: string;
  personaKey: string;
  privacyLevel: "public" | "internal" | "private" | "sensitive";
  scope: "persona_global" | "persona_user";
  tags?: string[];
  themes?: string[];
  timestamp?: Date | string | null;
  userId?: string | null;
};

export type HindsightRetainResult = {
  attempted: boolean;
  bankId?: string;
  enabled: boolean;
  error?: string;
  latencyMs?: number;
  operationId?: string | null;
  provider: "hindsight";
  skippedReason?: string;
  succeeded: boolean;
};

export type HindsightReflectInput = {
  context?: string | null;
  maxTokens?: number;
  organizationId: string;
  personaId: string;
  personaKey: string;
  query: string;
  scope: "persona_global" | "persona_user";
  tags?: string[];
  userId?: string | null;
};

export type HindsightReflectResult = {
  attempted: boolean;
  enabled: boolean;
  error?: string;
  latencyMs?: number;
  provider: "hindsight";
  skippedReason?: string;
  text: string | null;
};

export type HindsightPersonaMemoryClient = {
  recall(input: HindsightRecallInput): Promise<HindsightRecallResult>;
  reflect(input: HindsightReflectInput): Promise<HindsightReflectResult>;
  retain(input: HindsightRetainInput): Promise<HindsightRetainResult>;
};

type HindsightBudget = "low" | "mid" | "high";

type FetchLike = typeof fetch;

const DEFAULT_HINDSIGHT_CONFIG: HindsightPersonaMemoryConfig = {
  apiKey: null,
  baseUrl: null,
  enabled: false,
  failOpen: true,
  recallEnabled: false,
  recallTopK: 5,
  reflectEnabled: false,
  reflectTimeoutMs: 5000,
  retainAsyncEnabled: true,
  retainEnabled: false,
  timeoutMs: 1200,
};

function parseBoolean(
  value: boolean | string | undefined,
  defaultValue: boolean
): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return defaultValue;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return defaultValue;
}

function parsePositiveInteger(
  value: number | string | undefined,
  defaultValue: number
): number {
  const parsed =
    typeof value === "number"
      ? value
      : Number.parseInt(value?.trim() ?? "", 10);
  return Number.isFinite(parsed) && parsed > 0
    ? Math.floor(parsed)
    : defaultValue;
}

function normalizeBaseUrl(value: string | undefined): string | null {
  const trimmed = value?.trim();
  if (!trimmed) {
    return null;
  }
  return trimmed.replace(/\/+$/u, "");
}

export function createHindsightPersonaMemoryConfig(
  env?: HindsightPersonaMemoryEnv | null
): HindsightPersonaMemoryConfig {
  if (!env) {
    return DEFAULT_HINDSIGHT_CONFIG;
  }
  return {
    apiKey: env.HINDSIGHT_API_KEY?.trim() || null,
    baseUrl: normalizeBaseUrl(env.HINDSIGHT_BASE_URL),
    enabled: parseBoolean(env.HINDSIGHT_ENABLED, false),
    failOpen: parseBoolean(env.HINDSIGHT_FAIL_OPEN, true),
    recallEnabled: parseBoolean(env.HINDSIGHT_RECALL_ENABLED, false),
    recallTopK: parsePositiveInteger(env.HINDSIGHT_RECALL_TOP_K, 5),
    reflectEnabled: parseBoolean(env.HINDSIGHT_REFLECT_ENABLED, false),
    reflectTimeoutMs: parsePositiveInteger(
      env.HINDSIGHT_REFLECT_TIMEOUT_MS,
      5000
    ),
    retainAsyncEnabled: parseBoolean(env.HINDSIGHT_RETAIN_ASYNC_ENABLED, true),
    retainEnabled: parseBoolean(env.HINDSIGHT_RETAIN_ENABLED, false),
    timeoutMs: parsePositiveInteger(env.HINDSIGHT_TIMEOUT_MS, 1200),
  };
}

function createSkippedRecallResult(
  reason: string,
  enabled = false
): HindsightRecallResult {
  return {
    attempted: false,
    enabled,
    memories: [],
    provider: "hindsight",
    skippedReason: reason,
  };
}

function createSkippedRetainResult(
  reason: string,
  enabled = false
): HindsightRetainResult {
  return {
    attempted: false,
    enabled,
    provider: "hindsight",
    skippedReason: reason,
    succeeded: false,
  };
}

function createSkippedReflectResult(
  reason: string,
  enabled = false
): HindsightReflectResult {
  return {
    attempted: false,
    enabled,
    provider: "hindsight",
    skippedReason: reason,
    text: null,
  };
}

export function createNoopHindsightPersonaMemoryClient(
  reason = "disabled"
): HindsightPersonaMemoryClient {
  return {
    recall: async () => createSkippedRecallResult(reason),
    reflect: async () => createSkippedReflectResult(reason),
    retain: async () => createSkippedRetainResult(reason),
  };
}

function fnv1aBase36(input: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

function bankSegment(value: string): string {
  return fnv1aBase36(value || "empty");
}

export function hindsightPersonaBankId(input: {
  organizationId: string;
  personaId: string;
  scope: "persona_global" | "persona_user";
  userId?: string | null;
}): string {
  const org = bankSegment(input.organizationId);
  const persona = bankSegment(input.personaId);
  if (input.scope === "persona_global") {
    return `persona_${org}_${persona}`;
  }
  return `persona_user_${org}_${persona}_${bankSegment(input.userId ?? "")}`;
}

export function hindsightPersonaTags(input: {
  organizationId: string;
  personaId: string;
  themes?: string[];
  userId?: string | null;
}): string[] {
  return [
    `org_${bankSegment(input.organizationId)}`,
    `persona_${bankSegment(input.personaId)}`,
    ...(input.userId ? [`user_${bankSegment(input.userId)}`] : []),
    ...(input.themes ?? [])
      .map((theme) =>
        theme
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9_-]+/gu, "_")
      )
      .filter((theme) => theme.length > 0)
      .slice(0, 6)
      .map((theme) => `theme_${theme.slice(0, 48)}`),
  ];
}

function appendPath(baseUrl: string, path: string): string {
  return `${baseUrl}${path}`;
}

function withTimeoutSignal(timeoutMs: number): {
  cancel: () => void;
  signal: AbortSignal;
} {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  return {
    cancel: () => clearTimeout(timeout),
    signal: controller.signal,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function stringRecord(value: unknown): Record<string, string> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string"
    )
  );
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function toRecallMemory(
  bankId: string,
  value: unknown
): HindsightRawRecallMemory | null {
  if (!isRecord(value)) {
    return null;
  }
  const id = optionalString(value.id);
  const text = optionalString(value.text);
  if (!id || !text) {
    return null;
  }
  return {
    bankId,
    chunkId: optionalString(value.chunk_id ?? value.chunkId),
    context: optionalString(value.context),
    documentId: optionalString(value.document_id ?? value.documentId),
    entities: stringArray(value.entities),
    id,
    mentionedAt: optionalString(value.mentioned_at ?? value.mentionedAt),
    metadata: stringRecord(value.metadata),
    occurredEnd: optionalString(value.occurred_end ?? value.occurredEnd),
    occurredStart: optionalString(value.occurred_start ?? value.occurredStart),
    sourceFactIds: stringArray(value.source_fact_ids ?? value.sourceFactIds),
    tags: stringArray(value.tags),
    text,
    type: optionalString(value.type),
  };
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function recallQuery(input: HindsightRecallInput): string {
  return [
    input.message,
    input.retrievalQueries.episodic,
    input.retrievalQueries.semantic,
    input.retrievalQueries.source,
  ]
    .map((entry) => entry.trim())
    .filter(
      (entry, index, array) =>
        entry.length > 0 && array.indexOf(entry) === index
    )
    .join("\n");
}

async function requestJson(input: {
  apiKey: string | null;
  body: Record<string, unknown>;
  fetchImpl: FetchLike;
  timeoutMs: number;
  url: string;
}): Promise<unknown> {
  const timeout = withTimeoutSignal(input.timeoutMs);
  try {
    const response = await input.fetchImpl(input.url, {
      body: JSON.stringify(input.body),
      headers: {
        ...(input.apiKey ? { Authorization: `Bearer ${input.apiKey}` } : {}),
        "Content-Type": "application/json",
      },
      method: "POST",
      signal: timeout.signal,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(
        `Hindsight request failed with ${response.status}: ${text.slice(0, 300)}`
      );
    }
    return await response.json();
  } finally {
    timeout.cancel();
  }
}

function requiredCallConfig(
  config: HindsightPersonaMemoryConfig
): { baseUrl: string } | { skippedReason: string } {
  if (!config.enabled) {
    return { skippedReason: "disabled" };
  }
  if (!config.baseUrl) {
    return { skippedReason: "missing_base_url" };
  }
  return { baseUrl: config.baseUrl };
}

function handleError<T>(
  config: HindsightPersonaMemoryConfig,
  fallback: T,
  error: unknown,
  logger: PersonaLogger
): T {
  if (!config.failOpen) {
    throw error;
  }
  logger.warn(
    personaLogMessage("[persona-memory:hindsight] provider call failed:", error)
  );
  return fallback;
}

export function createHindsightPersonaMemoryClient(
  config: HindsightPersonaMemoryConfig,
  fetchImpl: FetchLike = fetch,
  logger: PersonaLogger = DEFAULT_PERSONA_LOGGER
): HindsightPersonaMemoryClient {
  return {
    async recall(input) {
      const callConfig = requiredCallConfig(config);
      if ("skippedReason" in callConfig) {
        return createSkippedRecallResult(
          callConfig.skippedReason,
          config.enabled
        );
      }
      if (!config.recallEnabled) {
        return createSkippedRecallResult("recall_disabled", config.enabled);
      }
      const query = recallQuery(input);
      if (!query) {
        return createSkippedRecallResult("empty_query", config.enabled);
      }
      const maxResults = Math.max(
        1,
        Math.min(input.maxResults, config.recallTopK)
      );

      const banks = [
        hindsightPersonaBankId({
          organizationId: input.organizationId,
          personaId: input.personaId,
          scope: "persona_user",
          userId: input.userId,
        }),
        hindsightPersonaBankId({
          organizationId: input.organizationId,
          personaId: input.personaId,
          scope: "persona_global",
        }),
      ];
      const startedAt = Date.now();
      try {
        const results = await Promise.all(
          banks.map(async (bankId) => {
            const response = await requestJson({
              apiKey: config.apiKey,
              body: {
                budget: "low" satisfies HindsightBudget,
                max_tokens: 2048,
                prefer_observations: true,
                query,
                query_timestamp: new Date().toISOString(),
                tags_match: "any",
                types: ["observation", "experience", "world"],
              },
              fetchImpl,
              timeoutMs: config.timeoutMs,
              url: appendPath(
                callConfig.baseUrl,
                `/v1/default/banks/${encodeURIComponent(bankId)}/memories/recall`
              ),
            });
            const memories =
              isRecord(response) && Array.isArray(response.results)
                ? response.results
                : [];
            return memories
              .map((memory) => toRecallMemory(bankId, memory))
              .filter(
                (memory): memory is HindsightRawRecallMemory => memory !== null
              );
          })
        );
        return {
          attempted: true,
          enabled: true,
          latencyMs: Date.now() - startedAt,
          memories: results.flat().slice(0, maxResults * 2),
          provider: "hindsight",
        };
      } catch (error) {
        return handleError(
          config,
          {
            attempted: true,
            enabled: true,
            latencyMs: Date.now() - startedAt,
            memories: [],
            provider: "hindsight",
            skippedReason: "recall_failed",
          },
          error,
          logger
        );
      }
    },

    async retain(input) {
      const callConfig = requiredCallConfig(config);
      if ("skippedReason" in callConfig) {
        return createSkippedRetainResult(
          callConfig.skippedReason,
          config.enabled
        );
      }
      if (!config.retainEnabled) {
        return createSkippedRetainResult("retain_disabled", config.enabled);
      }
      if (input.privacyLevel === "sensitive") {
        return createSkippedRetainResult("privacy_sensitive", config.enabled);
      }
      if (input.scope === "persona_global" && input.userId) {
        return createSkippedRetainResult(
          "user_specific_global_retain_denied",
          config.enabled
        );
      }

      const bankId = hindsightPersonaBankId({
        organizationId: input.organizationId,
        personaId: input.personaId,
        scope: input.scope,
        userId: input.userId,
      });
      const startedAt = Date.now();
      try {
        const response = await requestJson({
          apiKey: config.apiKey,
          body: {
            async: config.retainAsyncEnabled,
            items: [
              {
                content: input.content,
                context: input.context ?? "persona interaction memory",
                document_id: input.documentId ?? undefined,
                metadata: {
                  ...(input.chatMessageId
                    ? { chat_message_id: input.chatMessageId }
                    : {}),
                  ...(input.chatSessionId
                    ? { chat_session_id: input.chatSessionId }
                    : {}),
                  persona_key: input.personaKey,
                  privacy_level: input.privacyLevel,
                  provider_source: "persona_memory",
                  scope: input.scope,
                },
                observation_scopes: "shared",
                tags: [
                  ...hindsightPersonaTags({
                    organizationId: input.organizationId,
                    personaId: input.personaId,
                    themes: input.themes,
                    userId: input.userId,
                  }),
                  ...(input.tags ?? []),
                ],
                timestamp:
                  input.timestamp instanceof Date
                    ? input.timestamp.toISOString()
                    : (input.timestamp ?? undefined),
                update_mode: "append",
              },
            ],
          },
          fetchImpl,
          timeoutMs: config.timeoutMs,
          url: appendPath(
            callConfig.baseUrl,
            `/v1/default/banks/${encodeURIComponent(bankId)}/memories`
          ),
        });
        const operationId = isRecord(response)
          ? optionalString(response.operation_id ?? response.operationId)
          : null;
        return {
          attempted: true,
          bankId,
          enabled: true,
          latencyMs: Date.now() - startedAt,
          operationId,
          provider: "hindsight",
          succeeded: true,
        };
      } catch (error) {
        return handleError(
          config,
          {
            attempted: true,
            bankId,
            enabled: true,
            error: toErrorMessage(error),
            latencyMs: Date.now() - startedAt,
            provider: "hindsight",
            succeeded: false,
          },
          error,
          logger
        );
      }
    },

    async reflect(input) {
      const callConfig = requiredCallConfig(config);
      if ("skippedReason" in callConfig) {
        return createSkippedReflectResult(
          callConfig.skippedReason,
          config.enabled
        );
      }
      if (!config.reflectEnabled) {
        return createSkippedReflectResult("reflect_disabled", config.enabled);
      }
      const bankId = hindsightPersonaBankId({
        organizationId: input.organizationId,
        personaId: input.personaId,
        scope: input.scope,
        userId: input.userId,
      });
      const startedAt = Date.now();
      try {
        const response = await requestJson({
          apiKey: config.apiKey,
          body: {
            budget: "low" satisfies HindsightBudget,
            context: input.context ?? undefined,
            include: { facts: {} },
            max_tokens: input.maxTokens ?? 1200,
            query: input.query,
            tags: input.tags,
            tags_match: "any",
          },
          fetchImpl,
          timeoutMs: config.reflectTimeoutMs,
          url: appendPath(
            callConfig.baseUrl,
            `/v1/default/banks/${encodeURIComponent(bankId)}/reflect`
          ),
        });
        return {
          attempted: true,
          enabled: true,
          latencyMs: Date.now() - startedAt,
          provider: "hindsight",
          text: isRecord(response) ? optionalString(response.text) : null,
        };
      } catch (error) {
        return handleError(
          config,
          {
            attempted: true,
            enabled: true,
            error: toErrorMessage(error),
            latencyMs: Date.now() - startedAt,
            provider: "hindsight",
            text: null,
          },
          error,
          logger
        );
      }
    },
  };
}
