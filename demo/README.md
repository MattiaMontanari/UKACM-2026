# Prompt-cache live demo

Run this from the repository root after exporting your token:

```bash
export ZAI_API_TOKEN='...'
for script in demo/01-cold.sh demo/02-extend-history.sh demo/03-repeat.sh demo/04-change-early-prefix.sh; do
  bash "$script"
done
```

Each script prints the raw API JSON first and saves it under `demo/out/`. It then prints fields from the response:

```text
.usage.prompt_tokens
.usage.prompt_tokens_details.cached_tokens
.usage.completion_tokens
.usage.completion_tokens_details.reasoning_tokens
.usage.total_tokens
```

`uncached_prompt_tokens` is the script's calculation of `prompt_tokens - cached_tokens`; it is not an API field. The primary observation is the API's `cached_tokens` value. Providers may define any formal cache hit-rate separately.

## The four calls

1. `01-cold.sh`: first request.
2. `02-extend-history.sh`: re-sends the original messages, assistant answer, then a new question.
3. `03-repeat.sh`: same request payload as call 2.
4. `04-change-early-prefix.sh`: same as call 3 except the early system prompt changes `OpenFOAM v11` to `OpenFOAM v12`.

## Small edits to demonstrate cache behaviour

Edit only the final user question in `02-extend-history.sh`, then run it: the earlier prefix remains stable. Run `03-repeat.sh` again without editing: it sends the same payload. Change the early `v11` in `04-change-early-prefix.sh`: this tests whether an early change prevents reuse of the later prefix. Restore each edit before the next comparison.

The recorded lab observed cache tokens in 64-token increments, but thresholds, matching, cache lifetime, and cache scope are provider/model/time dependent. Treat the JSON response from your run as the result.

## Inspect without calling the API

```bash
bash tests/test-usage-summary.sh
for script in demo/[0-9][0-9]-*.sh; do DRY_RUN=1 bash "$script" | jq .; done
```
