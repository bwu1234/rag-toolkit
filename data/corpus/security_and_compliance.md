# Acme Cloud Storage — Security and Compliance

This document explains the security posture of ACS, including encryption, access control, auditability, and compliance controls.

## Encryption

ACS encrypts all object data at rest using AES-256 encryption. Keys are managed by the platform Key Management Service (KMS) and are rotated regularly according to the platform security policy.

All data in transit is encrypted using TLS 1.3. The Gateway enforces strong cipher suites and rejects weak or expired certificates.

## Access control

Access control is enforced at the Gateway and metadata layer:

- API keys are required for all requests.
- ACLs can be defined per object or per project.
- Gateway validates both the key scope and the requested operation.
- The metadata store also enforces project isolation and permission checks before object routing.

Role-based permissions are used for administrative and service-level operations. Only authorized platform administrators can perform purge, restore, or shard migration actions.

## Audit logging

ACS emits audit logs for security-sensitive actions:

- object creation, read, update, and delete
- metadata changes
- access-control changes
- restore and purge operations
- failed authentication and authorization attempts

Audit logs are forwarded to the central security log service and retained according to the platform’s retention policy.

## Compliance controls

ACS supports the following compliance controls:

- data retention policies for soft-delete and purge windows
- region-aware storage to keep data within approved jurisdictions
- customer-managed encryption keys for high-security workloads
- detailed audit trails for access and configuration changes
- immutable storage mode for regulatory workloads that disallow object overwrite or delete

## Incident handling

If a security incident is suspected:

1. Escalate immediately to the security incident response team.
2. Preserve logs and relevant metadata snapshots.
3. Quarantine affected objects or projects if necessary.
4. Follow the platform’s incident response process and document the timeline.

## Best practices

- Use project-scoped API keys rather than shared keys.
- Enable immutable mode for compliance-sensitive buckets.
- Restrict metadata tags to approved key-value patterns to avoid accidental exposure.
- Monitor audit logs for unusual access patterns.
