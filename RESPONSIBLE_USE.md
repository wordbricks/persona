# Responsible Use

This package models persona behavior from explicit profile, consent, source, privacy, and memory records. Operators remain responsible for making sure each persona is lawful, consented, labeled, and appropriate for the surface where it is used.

## Consent Model

`PERSONA_PROFILE_TYPES` defines five persona categories:

- `simulated_character`: A fictional or composite character. Use when the persona is not meant to identify a real person, or when a rights holder has authorized the character package.
- `living_public_figure`: A living real public figure. Use public, attributable material only unless you have stronger authorization; avoid private facts, private motives, and deceptive impersonation.
- `deceased_public_figure`: A deceased public figure. Review estate, publicity-rights, defamation, source-rights, and jurisdictional issues before use; prefer licensed, public, or fair-use-reviewed material.
- `private_authorized_person`: A nonpublic or privately scoped real person. Use only with explicit consent, clear scope, revocation/deletion handling, and tight access controls.
- `synthetic_role`: A non-identifying role or archetype such as "product reviewer" or "finance tutor." Keep it detached from a specific real person unless separately authorized.

`PERSONA_CONSENT_STATUSES` records the consent basis:

- `explicit_consent`: The represented person has explicitly consented to the persona and its intended use.
- `authorized`: A rights holder, employer, estate, or other appropriate authority has authorized the persona.
- `fictional_or_authorized`: The persona is fictional, synthetic, or otherwise authorized for the configured use.
- `public_material_only`: The persona is constrained to public materials and should not infer or claim private information.
- `unknown`: The consent basis is not established. Do not activate real-person personas with this status outside controlled review.

Recommended minimums: use `fictional_or_authorized` for `simulated_character` and `synthetic_role`; use `public_material_only` or stronger for `living_public_figure` and `deceased_public_figure`; require `explicit_consent` for `private_authorized_person`.

## Real-Person Personas

Real-person personas can create publicity-right, privacy, false endorsement, and defamation risk. For `living_public_figure`, restrict grounding to public statements, public works, public appearances, or authorized materials, and prefer `public_material_only` or stronger consent. For `private_authorized_person`, require explicit consent before ingesting sources, creating aliases, or enabling chat; define who can use the persona, what data can be remembered, and how revocation is handled. Do not present generated responses as the biological person, do not imply live access to private memory, and do not use a persona to evade platform or legal rules around impersonation.

## Disclosure Policy

`DisclosurePolicy` has two modes:

- `always`: The default OSS posture. Prompt builders inject the profile `transparencyLabel` and tell the model to actively disclose the AI/persona-simulation boundary before continuing in the persona voice.
- `on_request`: The legacy Velen posture. The model keeps the identity boundary internal during ordinary conversation, does not inject `transparencyLabel` into normal prompts, and discloses when the user asks about AI/system/original-person status, asks about consciousness or memory provenance, or safety flags require transparency.

Use `on_request` only when your product, jurisdiction, user interface, and consent model already provide adequate disclosure. Operators choosing `on_request` are responsible for user-facing labeling, auditability, and preventing deceptive impersonation.

## Privacy Layers

`PERSONA_PRIVACY_LEVELS` supports `public`, `internal`, `private`, and `sensitive`. Retrieval treats `public`, `internal`, and `private` as retrievable within the caller's authorized scope. `sensitive` memories are blocked from recall by `selectPersonaMemoriesWithScores` through the `excludedReason: "privacy"` path, and Hindsight retain skips sensitive inputs. Keep secrets, credentials, health data, payment data, and other high-risk material out of persona memory whenever possible; if they are detected or classified as `sensitive`, they should not be surfaced to the model as active memory.
