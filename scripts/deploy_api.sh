#!/usr/bin/env bash
# Zip and upload both Lambda handlers. Run after deploy_infra.sh.
source "$(dirname "$0")/_aws_common.sh"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

package() {  # $1 = api subdir, rest = extra files to bundle
  local name="$1"; shift
  mkdir -p "$BUILD/$name"
  cp "$REPO_ROOT/api/$name/handler.py" "$@" "$BUILD/$name/"
  # python's zipfile instead of `zip`, which Git Bash on Windows lacks
  (cd "$BUILD/$name" && "$PYTHON" -m zipfile -c "../$name.zip" ./*)
}

# match.csv lets intake reject unknown match_ids at the edge
package intake "$REPO_ROOT/data/shuttleset/set/match.csv"
package status

for name in intake status; do
  if [ "$name" = intake ]; then fn="$(stack_output IntakeFunctionName)"; else fn="$(stack_output StatusFunctionName)"; fi
  echo "Updating $fn"
  aws lambda update-function-code --function-name "$fn" \
    --zip-file "fileb://$BUILD/$name.zip" >/dev/null
  aws lambda wait function-updated --function-name "$fn"
done

echo "API: $(stack_output ApiUrl)"
