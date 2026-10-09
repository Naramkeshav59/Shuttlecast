#!/usr/bin/env bash
# Delete everything the stack created. The ALB alone bills ~$16/month
# whether or not anyone uses it, so tear down between demos.
source "$(dirname "$0")/_aws_common.sh"

read -r -p "Delete stack '$STACK' and ALL its data (results, videos, images)? [y/N] " ok
[ "$ok" = "y" ] || { echo "aborted"; exit 1; }

# Stop the services first: a worker mid-job writes its result to S3 after the
# bucket is emptied, and the stack delete then fails on a non-empty bucket.
CLUSTER="$(stack_output ClusterName)"
for key in WorkerServiceName UiServiceName; do
  aws ecs update-service --cluster "$CLUSTER" --service "$(stack_output "$key")" --desired-count 0 >/dev/null
done
aws ecs wait services-stable --cluster "$CLUSTER" \
  --services "$(stack_output WorkerServiceName)" "$(stack_output UiServiceName)"

# CloudFormation can't delete a non-empty bucket or ECR repo
BUCKET="$(stack_output BucketName)"
aws s3 rm "s3://$BUCKET" --recursive
# --force deletes the repo with its images. Deleting images one by one misses
# the attestation manifests Docker buildx pushes alongside each image, and
# the stack delete then fails on a "non-empty" repository.
for key in WorkerRepositoryName UiRepositoryName; do
  aws ecr delete-repository --repository-name "$(stack_output "$key")" --force >/dev/null
done

aws cloudformation delete-stack --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --stack-name "$STACK"
aws ssm delete-parameters --names /shuttlecast/groq_api_key /shuttlecast/gemini_api_key >/dev/null || true
echo "Stack $STACK deleted."
