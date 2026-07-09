import { describe, expect, it, vi } from "vitest";

import type { PersonaProfile } from "../schema";
import { upsertPersonaAlias } from "./profile";

const profile: PersonaProfile = {
  consentStatus: "fictional_or_authorized",
  createdAt: new Date("2026-01-01T00:00:00.000Z"),
  displayName: "Juno",
  id: "persona_1",
  organizationId: "org_1",
  personaKey: "juno",
  personaScope: "organization",
  personaType: "simulated_character",
  personaVersion: "v1",
  policy: {
    allowedUse: ["private_chat"],
    biographicalSummary: "Product visionary persona.",
    forbiddenUse: ["deceptive_impersonation"],
    knowledgeCutoffForPersona: "2024-01-01",
    transparencyLabel: "AI persona simulation for Juno.",
  },
  profile: {},
  sourceRef: null,
  state: "active",
  updatedAt: new Date("2026-01-01T00:00:00.000Z"),
  updatedByUserId: null,
};

class MockDbQuery<T> implements PromiseLike<T> {
  constructor(private readonly result: T) {}

  from(): this {
    return this;
  }

  innerJoin(): this {
    return this;
  }

  limit(): this {
    return this;
  }

  async then<TResult1 = T, TResult2 = never>(
    onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null
  ): Promise<TResult1 | TResult2> {
    return Promise.resolve(this.result).then(
      onfulfilled ?? undefined,
      onrejected ?? undefined
    );
  }

  where(): this {
    return this;
  }
}

function createPersonaAliasDbMock(selectResults: unknown[][]) {
  const pendingSelectResults = [...selectResults];
  return {
    insert: vi.fn(() => new MockDbQuery([])),
    select: vi.fn(() => new MockDbQuery(pendingSelectResults.shift() ?? [])),
  };
}

describe("upsertPersonaAlias", () => {
  it("does not silently move an existing alias to another persona", async () => {
    const db = createPersonaAliasDbMock([
      [profile],
      [],
      [{ personaId: "persona_2", personaKey: "mira" }],
    ]);

    await expect(
      upsertPersonaAlias(db as never, {
        aliasKey: "coach",
        organizationId: "org_1",
        personaKey: "juno",
        surface: "slack",
      })
    ).rejects.toThrow("Persona alias coach is already assigned to mira.");
    expect(db.insert).not.toHaveBeenCalled();
  });
});
