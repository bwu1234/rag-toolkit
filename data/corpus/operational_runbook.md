# Acme Cloud Storage — Operational Runbook

This runbook describes how to operate ACS in production, including monitoring, incident response, and escalation.

## Monitoring and alerting

Key metrics to monitor:

- Gateway request rate and error rate
- 429 rate and token-bucket saturation
- Shard latency for read and write operations
- Metadata store query latency and error rate
- Replica lag and replication throughput
- Disk capacity and I/O utilization per shard
- Soft-delete retention queue depth

Alert thresholds should be aligned with SLOs. For example:

- `Gateway error rate > 2%` sustained for 5 minutes
- `Shard read latency p95 > 500 ms`
- `Metadata store errors > 1%`
- `Replication degraded` on more than 5% of objects
- `Disk utilization > 80%` on any shard

## Health checks

ACS exposes a health endpoint at `GET /v1/health`.

A healthy response should include:

- Gateway connectivity status
- Metadata store reachable and nominal
- Shard manager healthy
- Replica count within tolerance

During incidents, use the health endpoint to determine whether the problem is isolated to the gateway, metadata store, or storage layer.

## Incident response

1. Triage the alert and identify the failing subsystem.
2. Check the health endpoint, recent logs, and deployment status.
3. If the Gateway is failing, verify the API key service and rate limit subsystem.
4. If reads or writes are failing, identify the affected shards and metadata store state.
5. If replication is degraded, confirm whether the issue is transient or due to disk/node failure.

Common incident categories:

- Gateway overload or misconfiguration
- Metadata store outage or slow queries
- Shard hardware failure or disk pressure
- Cross-region replication failures
- Data retention/purge job failures

## On-call procedures

- Use the `#acs-oncall` channel for urgent production issues.
- Document every incident in the platform ticketing system immediately.
- If a customer is impacted, acknowledge the incident and include expected recovery actions.
- For unclear failures, open a postmortem ticket and attach related logs and metrics.

## Recovery actions

### Gateway issues

- Restart the gateway process or rolling-deploy a fixed container.
- Verify that API key validation and routing are healthy.
- Use the load balancer to drain traffic from unhealthy instances.

### Shard issues

- Identify failing shard nodes and remove them from rotation if needed.
- Rebalance object ownership using the Shard Manager.
- If a node is disk-constrained, migrate hot objects to healthy nodes.

### Metadata store issues

- Check query execution plans for slow listing or metadata update operations.
- Clear any query caches if stale indexes are causing failures.
- Fail open only if metadata reads are non-critical and the action is safe for platform policy.

## Post-incident review

- Record the root cause and impact classification.
- Update runbook procedures if the incident exposed a missing tool or unclear step.
- Add a retrospective note to the on-call rotation so future teams benefit from the lessons learned.
