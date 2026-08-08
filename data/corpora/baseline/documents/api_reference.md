# Acme Cloud Storage — API Reference

This reference describes the core ACS API endpoints, authentication, error handling, and common usage patterns.

## Authentication

ACS uses API keys for authentication. Every request must include an `Authorization` header with the form:

```
Authorization: Bearer <api-key>
```

API keys are scoped to a single project and region. Keys may have read-only or read-write permissions depending on the team’s role and the platform policy.

## Endpoints

### Create object

`POST /v1/objects`

- Accepts multipart form data or raw binary payloads.
- Required headers: `Content-Type`, `X-ACS-Project`, `X-ACS-Region`.
- Optional metadata headers: `X-ACS-Meta-<key>`.
- Returns `201 Created` with a JSON body containing `object_id`, `location`, and `checksum`.

### Retrieve object

`GET /v1/objects/{object_id}`

- Supports range requests via `Range` headers.
- Returns `200 OK` with the object bytes and `Content-Type` preserved.
- For metadata-only reads, use `HEAD /v1/objects/{object_id}`.

### Delete object

`DELETE /v1/objects/{object_id}`

- Soft-deletes an object for 30 days by default.
- Returns `204 No Content` on success.
- The object can be restored within retention through the restore endpoint.

### Multipart upload

`POST /v1/objects/multipart`

- Initiate a multipart upload and receive an `upload_id`.
- Upload parts with `PUT /v1/objects/multipart/{upload_id}/parts/{part_number}`.
- Complete with `POST /v1/objects/multipart/{upload_id}/complete`.
- Abort with `DELETE /v1/objects/multipart/{upload_id}`.

Multipart is required for objects larger than 100 MB and recommended for improves reliability on high-latency networks.

### Metadata and query

`GET /v1/objects?project=<project>&prefix=<prefix>`

- Returns a paginated list of object metadata records.
- Supports filtering by key prefixes and metadata tags.
- Uses the metadata store rather than object shards for efficient list results.

### Health and diagnostics

`GET /v1/health`

- Returns service health status for the Gateway and downstream dependencies.
- Useful for monitoring readiness and for initial diagnostics during incidents.

## Error codes

- `400 Bad Request` — malformed request payload or missing required fields.
- `401 Unauthorized` — invalid or missing API key.
- `403 Forbidden` — key lacks permission for the requested operation.
- `404 Not Found` — object does not exist or is outside the requesting project.
- `409 Conflict` — multipart upload conflict or duplicate object key.
- `429 Too Many Requests` — rate limit exceeded.
- `500 Internal Server Error` — unexpected failure in a downstream shard or metadata store.

## Rate limits and retry behavior

Rate limits are enforced at the Gateway using a token bucket. The default quota is 1,000 requests per minute per API key. Clients should honor `Retry-After` headers on `429` responses and use exponential backoff for retries.

## Best practices

- Use multipart upload for objects larger than 100 MB.
- Cache object metadata locally when possible; list operations are cheaper than repeated metadata reads.
- Respect rate limits and avoid retry storms by backing off on `429` responses.
- Store the returned object `checksum` and compare it with the final response to verify data integrity.
