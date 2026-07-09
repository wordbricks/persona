import type { PostgresJsDatabase } from "drizzle-orm/postgres-js";

export type PersonaDatabase = PostgresJsDatabase<Record<string, unknown>>;
