import { describe, expect, it } from "vitest";

import {
  PERSONA_CONSOLIDATION_SYSTEM_PROMPT,
  PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT,
  PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT,
  PERSONA_TURN_PLANNER_SYSTEM_PROMPT,
} from "./prompts";
import { personaSourceDraftMemorySchema } from "./source-ingestion";
import {
  consolidationOutputSchema,
  PERSONA_ANSWER_MODES,
  personaTurnPlanSchema,
} from "./types";

function promptJsonExample(prompt: string): unknown {
  const marker = "Return exactly this shape: ";
  const lines = prompt.split("\n");
  const index = lines.findIndex(
    (entry) => entry === marker.trimEnd() || entry.startsWith(marker)
  );
  if (index < 0) {
    throw new Error("Missing prompt JSON example.");
  }
  const inlineJson = lines[index]?.slice(marker.length);
  return JSON.parse(inlineJson || lines[index + 1] || "");
}

describe("persona turn planner prompt", () => {
  it("shows an example accepted by the exported workflow schema", () => {
    expect(
      personaTurnPlanSchema.safeParse(
        promptJsonExample(PERSONA_TURN_PLANNER_SYSTEM_PROMPT)
      ).success
    ).toBe(true);
  });

  it("only exposes the supported answer modes", () => {
    expect(PERSONA_ANSWER_MODES).toEqual([
      "persona_grounded_response",
      "uncertain_inference",
      "transparent_meta_response",
    ]);
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "- persona_grounded_response"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "- uncertain_inference"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "- transparent_meta_response"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).not.toContain(
      "source_grounded_summary"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).not.toContain(
      "out_of_scope_refusal"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).not.toContain("safety_refusal");
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).not.toContain(
      "style_only_transformation"
    );
  });

  it("routes concrete item and brand questions to source-backed retrieval", () => {
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "concrete products, brands"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain("persona_fact");
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain("source_chunk");
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "do not use transparent_meta_response merely because the detail is personal"
    );
  });

  it("documents queryFocus time as a single nullable value", () => {
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "gate.queryFocus.time must be one string or null, never an array"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "context.contextKeys.timePeriods is the array for multiple time periods"
    );
  });

  it("documents that memory routing can be empty for tool smoke tests", () => {
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "[] when the turn can be answered without pre-activating durable persona memory"
    );
    expect(PERSONA_TURN_PLANNER_SYSTEM_PROMPT).toContain(
      "safe tool smoke tests"
    );
  });
});

describe("persona consolidation prompt", () => {
  it("shows a schema-complete JSON example", () => {
    const example = promptJsonExample(PERSONA_CONSOLIDATION_SYSTEM_PROMPT) as {
      beliefs: unknown[];
      episodes: unknown[];
      habits: unknown[];
      styleProfiles: unknown[];
    };
    expect(consolidationOutputSchema.safeParse(example).success).toBe(true);
    expect(example.beliefs[0]).toMatchObject({
      action: "create",
      beliefType: "professional_norm",
      confidence: expect.any(Number),
      domain: expect.any(String),
      proposition: expect.any(String),
      strength: expect.any(Number),
    });
    expect(example.episodes[0]).toMatchObject({
      confidence: expect.any(Number),
      eventSummary: expect.any(String),
      title: expect.any(String),
    });
    expect(example.habits[0]).toMatchObject({
      confidence: expect.any(Number),
      defaultResponsePattern: expect.any(Array),
      strength: expect.any(Number),
      trigger: {
        description: expect.any(String),
        type: expect.any(String),
      },
    });
    expect(example.styleProfiles[0]).toMatchObject({
      register: expect.any(String),
      sentenceLength: expect.any(String),
      toneVector: expect.any(Object),
    });
  });
});

describe("persona source drafting prompt", () => {
  it("shows an example accepted by the exported workflow schema", () => {
    expect(
      personaSourceDraftMemorySchema.safeParse(
        promptJsonExample(PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT)
      ).success
    ).toBe(true);
  });
});

describe("persona source drafting prompt", () => {
  it("asks the source drafter to preserve concrete source-backed facts", () => {
    expect(PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT).toContain("sourceFacts");
    expect(PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT).toContain("toothbrush brand");
    expect(PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT).toContain(
      "products, brands, tools"
    );

    const marker = "Return exactly this shape: ";
    const line = PERSONA_SOURCE_DRAFTING_SYSTEM_PROMPT.split("\n").find(
      (entry) => entry.startsWith(marker)
    );
    expect(line).toBeTruthy();
    if (!line) {
      throw new Error("Missing source drafting prompt JSON example.");
    }
    const example = JSON.parse(line.slice(marker.length));
    expect(example.sourceFacts[0]).toMatchObject({
      claimText: expect.stringContaining("toothbrush"),
      confidence: expect.any(Number),
      evidenceSpan: expect.any(String),
      factType: "uses_product",
      firstPersonForm: expect.any(String),
      objectName: expect.stringContaining("toothbrush"),
      objectType: "product",
      sourceChunkIds: expect.any(Array),
    });
  });
});

describe("persona memory triage prompt", () => {
  it("documents the LLM triage prompt output contract", () => {
    expect(PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT).toContain("shouldRemember");
    expect(PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT).toContain("privacyLevel");
    expect(PERSONA_MEMORY_TRIAGE_SYSTEM_PROMPT).toContain("dedupeKey");
  });
});
