#!/usr/bin/env bash
# Build + push the worker and UI images to ECR, then roll both ECS services
# onto them. Run after deploy_infra.sh; needs Docker running.
#
# Images are tagged with the git SHA, not :latest -- a new tag is what makes
# CloudFormation register a new task definition and ECS roll it out, and it
# makes "what's running in prod" and rollbacks unambiguous.
source "$(dirname "$0")/_aws_common.sh"

TAG="${IMAGE_TAG:-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"

aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"

for svc in worker ui; do
  if [ "$svc" = worker ]; then repo="$(stack_output WorkerRepositoryName)"; else repo="$(stack_output UiRepositoryName)"; fi
  image="$REGISTRY/$repo:$TAG"
  echo "Building $image"
  docker build --platform linux/amd64 -f "$(native "$REPO_ROOT/$svc/Dockerfile")" -t "$image" "$(native "$REPO_ROOT")"
  docker push "$image"
done

WORKER_COUNT="${WORKER_COUNT:-1}" UI_COUNT="${UI_COUNT:-1}" IMAGE_TAG="$TAG" \
  "$SCRIPT_DIR/deploy_infra.sh"

echo "UI: $(stack_output UiUrl)  (allow ~2 min for the first task to pass ALB health checks)"
