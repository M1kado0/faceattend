# PostgreSQL/FAISS to Local SQLite Migration Safety Plan

**Status:** Safety procedure defined; execution remains proposed and requires approval.

## Purpose

Move reusable attendance identities and compatible embeddings into the local
desktop store without treating legacy crawler/search records as attendance data,
mixing incompatible model versions, or deleting the source systems prematurely.

## Source-of-truth boundaries

- PostgreSQL `users`, `face_registrations`, `attendance_sessions`,
  `attendance_records`, and `audit_log` provide legacy metadata.
- The FAISS `.index` plus matching `.meta.json` must be read together. Neither
  file alone proves which vector belongs to which registration.
- `face_registrations.embedding_id` must resolve to exactly one FAISS `id_map`
  entry and one metadata entry. Missing, duplicate, or reordered mappings are
  quarantined rather than guessed.
- Legacy crawler, cluster, match-monitoring, takedown, billing, and notification
  records are outside the local attendance import.

## Preflight and immutable export

1. Stop writes or take one transactionally consistent PostgreSQL snapshot.
2. Copy the FAISS index and metadata sidecar together while writes are stopped.
3. Record source commit, schema revision, row counts, file sizes, and SHA-256
   checksums. Work only from copies; keep originals read-only.
4. Reject the import if FAISS `ntotal`, `id_map` length, and relevant metadata
   count differ, or if any ID is duplicated.
5. Produce a dry-run manifest containing only IDs, mapping status, model metadata,
   vector dimension/norm status, and proposed destination IDs. Never put vectors,
   face crops, frames, or credentials in this report.

## Mapping and compatibility rules

| Legacy value | Local value | Rule |
|---|---|---|
| `users.id` | `people.id` | Import only explicitly consented attendance identities; do not infer consent. |
| `face_registrations` | `enrollment_sessions` + `embedding_templates` | Preserve source ID in audit metadata; synthesize no successful enrollment without verified liveness/consent evidence. |
| FAISS vector | `embedding_templates.embedding` | Require finite 512D float32, L2-normalize a copy only when source normalization is documented, and record that transformation. Otherwise quarantine. |
| `embedding_model_version` | `model_versions` | Require exact name, version, checksum, dimension, and preprocessing contract. A version label alone is insufficient. |
| `attendance_sessions` | `attendance_sessions` | Import only records that have unambiguous attendance semantics. |
| `attendance_records` | `attendance_records` | Do not import as verified local check-ins unless mandatory active/passive liveness evidence can be linked. Preserve unverifiable rows in an external archival export. |
| `audit_log` | `audit_events` | Preserve append-only history and sanitize metadata; do not copy IP/user-agent fields unless justified by the local retention policy. |

## Staged import

1. Create a new empty SQLite database using numbered local migrations.
2. Import configuration/model versions first, then people and explicit consent,
   enrollment metadata, compatible templates, sessions, verified attendance, and
   sanitized audit records in foreign-key order.
3. Use one transaction per bounded batch. On any row error, roll back that batch
   and write a non-biometric quarantine reason to the dry-run report.
4. Load templates through `load_compatible_templates()` and rebuild an
   `ExactNumpyMatcher`; do not copy the FAISS index into the new runtime.
5. Compare source/destination counts, stable IDs, model groups, and vector
   checksums. Run genuine/impostor score parity on an approved local evaluation
   set; never use production attendance rows as ground truth by assumption.
6. Restart the desktop process and verify reload, unknown rejection, ambiguity
   rejection, duplicate attendance, template erasure, and audit append-only rules.

## Cutover and rollback

- Run desktop SQLite in shadow/read-only comparison first. Do not dual-write until
  an ADR defines conflict ownership.
- Keep PostgreSQL/FAISS unchanged and recoverable until parity is reviewed and the
  user explicitly approves retirement.
- A failed import is discarded as a new SQLite file; never “repair” the source in
  place.
- Do not delete old services, source exports, or incompatible templates as part of
  migration. Record rejected rows and the exact reason.

## Decisions still required

- Which legacy identities have valid explicit biometric-attendance consent.
- Whether any legacy attendance row has linkable active and passive liveness
  evidence strong enough to import as verified attendance.
- Trusted embedding model identity/checksum and whether legacy vectors used the
  same preprocessing contract.
- Retention and deletion policy for the immutable migration snapshot.
