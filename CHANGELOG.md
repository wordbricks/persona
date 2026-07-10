# Changelog

## 0.2.0 - 2026-07-10

### Breaking changes

Persona ownership scopes now use tenant terminology throughout the public API,
Drizzle schema, persisted scope values, index names, examples, and Hindsight
metadata.

- `organizationId` is now `tenantId`.
- `sourceOrganizationId` is now `sourceTenantId`.
- `targetOrganizationId` is now `targetTenantId`.
- PostgreSQL columns named `organization_id` are now `tenant_id`.
- `PersonaScope` and `PERSONA_SCOPES` use `"tenant"` instead of
  `"organization"`.
- Hindsight tags use the `tenant_` prefix instead of `org_`.
- The `"organization"` member of `PersonaFactObjectType` is unchanged because
  it classifies a fact's subject and is unrelated to tenancy.

No deprecated aliases are provided. Consumers must update their code and data
before installing 0.2.0.

### Consumer migration checklist

1. Update application call sites, direct Drizzle queries, inferred row types,
   queued jobs, cached values, and serialized events to use `tenantId`.
2. Rename `organization_id` to `tenant_id` in these tables:

   - `persona_profiles`
   - `persona_aliases`
   - `persona_source_documents`
   - `persona_source_chunks`
   - `persona_facts`
   - `persona_episode_memories`
   - `persona_emotional_salience`
   - `persona_mood_states`
   - `persona_semantic_beliefs`
   - `persona_habit_patterns`
   - `persona_style_profiles`
   - `persona_workspace_states`
   - `persona_interaction_memories`
   - `persona_external_memory_refs`
   - `persona_lifecycle_effects`
   - `persona_memory_embeddings`
   - `persona_audit_logs`

3. Update the persona profile scope data and default:

   ```sql
   UPDATE persona_profiles
   SET persona_scope = 'tenant'
   WHERE persona_scope = 'organization';

   ALTER TABLE persona_profiles
     ALTER COLUMN persona_scope SET DEFAULT 'tenant';
   ```

4. Rename these indexes after renaming their columns:

   - `idx_persona_profiles_org_key` to
     `idx_persona_profiles_tenant_key`
   - `idx_persona_profiles_org_state` to
     `idx_persona_profiles_tenant_state`
   - `idx_persona_aliases_org_surface_state` to
     `idx_persona_aliases_tenant_surface_state`
   - `idx_persona_aliases_org_surface_alias` to
     `idx_persona_aliases_tenant_surface_alias`

5. Update Hindsight tag filters and stored tag references from `org_*` to
   `tenant_*`. Hindsight bank IDs remain stable if the identifier value passed
   as `tenantId` is the same value previously passed as `organizationId`.
6. Coordinate the database migration with every application instance that uses
   this package. Versions 0.1.x and 0.2.0 cannot safely share the same schema
   during a rolling deployment.

Review generated migrations carefully. A tool may interpret the renamed columns
as drop-and-add operations, which can destroy existing data.
