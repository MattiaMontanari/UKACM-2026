#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
response="$script_dir/out/03-repeat.json"
mkdir -p "$script_dir/out"

payload=$(cat <<'JSON'
{
  "model": "glm-5.3",
  "reasoning_effort": "low",
  "thinking": { "type": "disabled" },
  "messages": [
    { "role": "system", "content": "You serve as a CFD simulation engineer for OpenFOAM v11 case setup. Keep every reply under three sentences. Quote dictionary keywords exactly as they appear in the v11 user guide, including defaults. Flag explicitly whenever a question refers to another OpenFOAM release. Admit ignorance rather than guessing." },
    { "role": "user", "content": "What is the first parameter in blockMeshDict and what does it do?" },
    { "role": "assistant", "content": "The first entry is `convertToMeters`, a scaling factor that converts the vertex coordinates (specified in the units of the geometry file) into metres. For example, `convertToMeters 0.001;` treats coordinates as millimetres, converting them to SI units for the mesh." },
    { "role": "user", "content": "What is its default value?" }
  ]
}
JSON
)

if [[ "${DRY_RUN:-}" == "1" ]]; then printf '%s\n' "$payload"; exit 0; fi
if [[ -z "${ZAI_API_TOKEN:-}" ]]; then printf 'ZAI_API_TOKEN must be set.\n' >&2; exit 1; fi

curl --fail-with-body --silent --show-error https://api.z.ai/api/coding/paas/v4/chat/completions \
  -H "Authorization: Bearer $ZAI_API_TOKEN" -H 'Content-Type: application/json' \
  --data "$payload" | tee "$response"
printf '\nUsage summary (%s):\n' "$response"
jq -f "$script_dir/usage-summary.jq" "$response"
