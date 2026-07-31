# Acme Cloud Storage — Performance and Scaling

This document describes how ACS is designed to scale, what performance constraints exist, and how the platform handles load.

## Scaling architecture

ACS is designed to scale horizontally:

- Gateways are stateless and can be added or removed behind a load balancer.
- Storage shards own object key ranges and scale by adding new shard nodes or whole shard clusters.
- The Shard Manager coordinates shard membership and consistent hashing to rebalance traffic.
- The metadata store is separately scaled for read-heavy listing workloads and write-heavy update workloads.

Separation of object bytes from metadata allows each layer to scale independently.

## Throughput and latency

Typical performance targets:

- `GET` latency: under 200 ms for cached shard reads
- `PUT` latency: under 300 ms for small objects
- `p95` list latency: under 400 ms for metadata queries
- sustained throughput: 1,000 requests per second per region for general traffic

Performance depends on the object size, network latency, and shard load. Multipart upload improves throughput for large objects by parallelizing part uploads.

## Caching and read optimization

The Gateway uses sharded caches for hot object location lookups and metadata query results. Caches are kept short-lived to preserve consistency but reduce load on the Shard Manager and metadata store.

Read optimization strategies:

- object location caching at the Gateway
- warm-up of popular shards before traffic spikes
- prioritized read scheduling for latency-sensitive requests

## Shard rebalancing

Shard rebalancing is controlled by the Shard Manager and is triggered when:

- a new shard is added
- a shard is removed for maintenance
- disk occupancy exceeds the configured threshold
- the object key distribution becomes uneven

Rebalancing is performed incrementally to avoid large collateral load spikes. Objects are migrated in the background while requests continue to be served.

## Capacity planning

Capacity planning should account for:

- average object size and request type mix
- expected growth in write throughput
- regional replication overhead
- disk utilization headroom for rebalancing and failure recovery

A healthy shard cluster maintains at least 20% free capacity to absorb sudden growth and failed node rebuilds.

## Performance troubleshooting

Key areas to investigate when performance degrades:

- Gateway CPU or network saturation
- metadata store slow queries or lock contention
- shard disk pressure or I/O queue depth
- failed or slow replication causing degraded status
- rate-limit throttling causing client retries

Use coordinated tracing and metrics from the Gateway, Shard Manager, and metadata store to pinpoint the bottleneck.
