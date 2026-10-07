#!/usr/bin/env bash
# Create/update the CloudFormation stack (infra/template.yaml).
#
#   scripts/deploy_infra.sh                       # first run: services at 0 tasks
#   WORKER_COUNT=1 UI_COUNT=1 IMAGE_TAG=abc123 scripts/deploy_infra.sh
#
# Parameters you don't pass keep their previous values, so re-running this
# never silently scales running services back to zero.
source "$(dirname "$0")/_aws_common.sh"

: "${GROQ_API_KEY:?GROQ_API_KEY must be set in .env}"

# API keys go to SSM as SecureString -- never into the template, the image,
# or plain task-definition env vars.
aws ssm put-parameter --name /shuttlecast/groq_api_key --type SecureString \
  --value "$GROQ_API_KEY" --overwrite >/dev/null
overrides=()
if [ -n "${GEMINI_API_KEY:-}" ]; then
  aws ssm put-parameter --name /shuttlecast/gemini_api_key --type SecureString \
    --value "$GEMINI_API_KEY" --overwrite >/dev/null
  overrides+=("GeminiApiKeyParam=/shuttlecast/gemini_api_key")
fi

VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)"
SUBNETS="$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" Name=default-for-az,Values=true \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')"
overrides+=("VpcId=$VPC_ID" "SubnetIds=$SUBNETS")

[ -n "${WORKER_COUNT:-}" ] && overrides+=("WorkerDesiredCount=$WORKER_COUNT")
[ -n "${UI_COUNT:-}" ] && overrides+=("UiDesiredCount=$UI_COUNT")
[ -n "${IMAGE_TAG:-}" ] && overrides+=("ImageTag=$IMAGE_TAG")
[ -n "${MAX_RALLIES:-}" ] && overrides+=("MaxRallies=$MAX_RALLIES")

echo "Deploying stack $STACK to $AWS_DEFAULT_REGION (VPC $VPC_ID)..."
aws cloudformation deploy \
  --stack-name "$STACK" \
  --template-file "$REPO_ROOT/infra/template.yaml" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides "${overrides[@]}"

aws cloudformation describe-stacks --stack-name "$STACK" \
  --query 'Stacks[0].Outputs' --output table
