#!/usr/bin/env bash
# Shared setup for the deploy scripts. Source it, don't run it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/scripts"
STACK="${STACK_NAME:-shuttlecast}"
PYTHON="${PYTHON:-python3}"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
# .env ships with empty AWS_* lines; exported empty values would override
# credentials from `aws configure`, so drop them.
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION; do
  if [ -z "${!var:-}" ]; then unset "$var"; fi
done
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

command -v aws >/dev/null || { echo "aws CLI not found on PATH" >&2; exit 1; }

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
