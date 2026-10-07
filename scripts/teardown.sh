#!/usr/bin/env bash
# Delete everything the stack created. The ALB alone bills ~$16/month
# whether or not anyone uses it, so tear down between demos.
source "$(dirname "$0")/_aws_common.sh"

read -r -p "Delete stack '$STACK' and ALL its data (results, videos, images)? [y/N] " ok
[ "$ok" = "y" ] || { echo "aborted"; exit 1; }

# CloudFormation can't delete a non-empty bucket or ECR repo
BUCKET="$(stack_output BucketName)"
aws s3 rm "s3://$BUCKET" --recursive
for key in WorkerRepositoryName UiRepositoryName; do
  repo="$(stack_output "$key")"
  ids="$(aws ecr list-images --repository-name "$repo" --query 'imageIds' --output json)"
  if [ "$ids" != "[]" ]; then
    aws ecr batch-delete-image --repository-name "$repo" --image-ids "$ids" >/dev/null
  fi
done

aws cloudformation delete-stack --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --stack-name "$STACK"
aws ssm delete-parameters --names /shuttlecast/groq_api_key /shuttlecast/gemini_api_key >/dev/null || true
echo "Stack $STACK deleted."
