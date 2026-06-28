# WP-00 Containment Operator Runbook

This runbook applies the repository-side WP-00 controls. Position disposition,
token rotation, service restarts, and every other production mutation require an
approved maintenance window. Do not clear pauses or alter retained incident state
as part of this procedure.

## Preconditions

- Record the approved window, operator, reviewer, image digest, and rollback digest.
- Confirm affected trading remains paused and incident evidence is read-only.
- Set `MODEL_EVAL_AUTO_PROMOTE=0` in deployment policy and retain a copy of the
  current evaluator control file for evidence.

## Redis Secret And Startup

1. Create a password containing only letters, digits, underscore, and hyphen. Write
   it directly to an operator-managed file without echoing or printing the value.
2. Restrict the file to its owner (`chmod 600` on Ubuntu), and keep it outside the
   repository and Docker build context.
3. Set `REDIS_PASSWORD_FILE` to the absolute file path. Optionally set
   `REDIS_ACL_USERNAME`; it follows the same nonempty character allowlist.
4. Run `docker compose config --quiet`. Inspect rendered configuration for no Redis
   `ports` publication and no secret value. Do not paste rendered secrets into logs.
5. Start Redis in the approved window, then start dependent services only after its
   authenticated health check reports healthy.
6. From an unapproved external host/interface, verify TCP port 6379 is unreachable.
   From the Compose network, verify unauthenticated commands fail and the
   secret-backed authenticated health probe returns `PONG`.

The current Compose application services do not consume Redis credentials directly.
Before adding an authenticated Redis execution client, implement reviewed secret-file
loading in that client, grant only its required key/channel patterns and commands,
mount the secret only into that service, and add authenticated integration tests.

## Telegram Rotation And Log Verification

1. Deploy the logging suppression and redaction code before rotating the token.
2. Rotate the token through Telegram's authorized administration flow; never print,
   paste into command history, or record either token in tickets or logs.
3. Update the operator-managed deployment secret and restart only the bot service.
4. Verify the old token fails and the new bot identity responds as expected.
5. Exercise a canary token-shaped value through request and exception logging. Search
   new logs for the raw value, percent-encoded value, `/bot.../` URL segment, and URL
   user-info. All searches must return no credential-bearing output.
6. Quarantine retained affected logs, restrict access, record hashes and retention
   decisions, and securely expire copies under the approved incident policy.

## Evaluator Verification

1. Confirm the deployment environment resolves `MODEL_EVAL_AUTO_PROMOTE=0`.
2. Place `auto_promote=true` in a disposable control file and run the evaluator
   startup test. Effective auto-promotion must remain false.
3. Confirm the administrative command reports policy-disabled rather than enabled.
4. Do not modify the production registry, active pointer, or quarantine records.

## Rollback

1. Stop affected services without clearing pauses or changing position state.
2. Redeploy the recorded prior image digest and prior reviewed Compose definition.
3. Keep the rotated Telegram token; never restore a compromised credential.
4. If Redis authentication prevents startup, keep Redis private and services stopped;
   do not restore public unauthenticated access. Correct the secret/ACL configuration
   under the maintenance window and repeat validation.
5. Record outcomes, timestamps, health evidence, and rollback reason without secret
   values. Escalate any unexpected position or registry mutation immediately.
