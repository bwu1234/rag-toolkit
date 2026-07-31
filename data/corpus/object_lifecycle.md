# Acme Cloud Storage — Object Lifecycle

This document describes the lifecycle of objects in ACS, from creation through soft delete, restore, and permanent purge.

## Object creation

When a client uploads an object, the Gateway authenticates the request and forwards it to the appropriate storage shard. The Gateway also writes a metadata record to the Metadata Store before acknowledging the upload, so listings and access control can work independently of the object bytes.

The upload response includes:

- `object_id`: a stable identifier for the stored object
- `location`: the shard or bucket path where the object resides
- `checksum`: the computed object checksum for integrity verification
- `version`: the initial version token for the object

## Metadata model

Each object is represented by a metadata entry containing:

- `object_id`
- `project_id`
- `region`
- `size`
- `content_type`
- `checksums` (SHA-256 and optional user-provided digest)
- `created_at`
- `updated_at`
- `acl` / access-control entries
- `tags` and user-defined metadata fields
- `storage_class`
- `replication_status`
- `soft_delete_until`

The metadata store is the authoritative source for object properties, and it supports fast listing queries. Object bytes on shards are ultimately authoritative for read operations, but the metadata layer is consulted first for routing and permission checks.

## Soft delete

Deleting an object moves it into a soft-delete state for 30 days. During that period:

- the object remains recoverable via restore operations
- reads are blocked unless the requester is system-admin or on-call maintenance
- the metadata still retains the original object properties plus a `deleted_at` timestamp

Soft delete protects against accidental deletions and serves regulatory retention requirements.

## Restore

A deleted object can be restored with `POST /v1/objects/{object_id}/restore` within the retention window. Restore resets the `deleted_at` state and returns the object to normal visibility.

Restores must verify that the object's shard still has the data available. If the shard has already reclaimed the storage for that object, the restore fails with `410 Gone` and an incident ticket should be opened.

## Permanent purge

After 30 days, soft-deleted objects are permanently purged from both the metadata store and every storage shard replica. Permanent purge is irreversible.

The purge process runs asynchronously in the background and is gated by a retention job scheduler. It emits audit records for every object removed.

## Versioning and immutability

ACS supports immutable objects and versioning for compliance-sensitive workloads. When versioning is enabled for a project:

- each write creates a new version record
- old versions remain readable unless explicitly purged
- object IDs remain stable while version tokens advance

Immutable mode prevents overwrite or delete operations unless a higher privilege system user performs an exception workflow.

## Replication status

Replication status is reported as part of object metadata. It can be one of:

- `pending` — the object has been written to the primary shard and is still being replicated
- `replicated` — at least three replicas are fully committed
- `degraded` — one or more replicas are unavailable or lagging
- `failed` — replication did not complete successfully and the object requires manual recovery

Monitoring replication status is essential for service-level objectives and data durability.

## Object lifecycle events

ACS emits lifecycle events to the internal event bus for:

- object creation
- metadata updates
- state transitions (`deleted`, `restored`, `purged`)
- replication failures
- access control changes

These events are consumed by auditing, alerts, and downstream analytics.
