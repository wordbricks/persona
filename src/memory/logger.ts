import type { PersonaLogger } from "./types";

const runtimeConsole = globalThis.console;

export const DEFAULT_PERSONA_LOGGER: PersonaLogger = {
  error(message) {
    runtimeConsole.error(message);
  },
  warn(message) {
    runtimeConsole.warn(message);
  },
};

export function personaLogMessage(message: string, error: unknown): string {
  return `${message} ${error instanceof Error ? error.message : String(error)}`;
}
