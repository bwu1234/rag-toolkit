# Acme Cloud Storage — Architecture Overview

Acme Cloud Storage (ACS) is an internal object storage platform used by
product teams to store and serve user-uploaded files. This document describes
the major components and how they fit together.

## Components

### Gateway

The Gateway is a stateless HTTP service that authenticates requests, applies
rate limits, and routes them to the appropriate backend shard. It is the only
component exposed outside the internal network. Gateways are deployed behind
a load balancer in each region and scale horizontally based on request volume.

### Shard Manager

The Shard Manager keeps track of which storage shard owns which object key. It
maintains a consistent-hashing ring and coordinates rebalancing when shards
are added or removed. Clients never talk to the Shard Manager directly — the
Gateway consults it on every request and caches the result for a short TTL.

### Storage Shards

Each storage shard is a cluster of nodes that hold object data on local disks,
replicated three ways for durability. Shards expose a simple internal API for
get, put, and delete operations and report health metrics to the Shard
Manager every few seconds.

### Metadata Store

Object metadata (size, content type, checksums, access-control entries,
creation time) lives in a separate metadata store backed by a distributed SQL
database. Keeping metadata separate from object bytes lets the system scale
each independently and makes metadata queries fast without touching shard
storage.

## Request flow

1. A client sends a request to the Gateway with an API key.
2. The Gateway authenticates the key and checks rate limits.
3. The Gateway asks the Shard Manager which shard owns the object key.
4. The Gateway forwards the request to that shard and streams the response
   back to the client.
5. For writes, the Gateway also updates the Metadata Store once the shard
   confirms the write succeeded.

## Replication and durability

Every object is written to three replicas across at least two availability
zones before a write is acknowledged to the client. If a replica becomes
unavailable, the Shard Manager schedules a re-replication job to restore the
target replica count. ACS targets eleven nines of durability and four nines
of availability per region.

## Rate limiting

Rate limits are enforced at the Gateway using a token-bucket algorithm keyed
by API key. The default limit is 1,000 requests per minute per key, and teams
can request higher limits through the platform on-call channel.
