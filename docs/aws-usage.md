# AWS usage

## Batch jobs

```sh
# Submit EarthCARE retry5 and monitor
$ EC_JOB=$(aws batch submit-job \
  --job-name join-virtualize-earthcare-prod-retry5 \
  --job-queue join-virtualize \
  --job-definition join-virtualize-earthcare \
  --container-overrides '{"command": ["python", "scripts/earthcare_virtualize.py", "--execution-type", "prod"]}' \
  --region us-west-2 --query 'jobId' --output text)

echo "EarthCARE retry5: $EC_JOB"

while true; do
  OUTPUT=$(aws batch describe-jobs --jobs $EC_JOB --region us-west-2 \
    --query 'jobs[0].status' --output text 2>&1)
  if echo "$OUTPUT" | grep -qi "ExpiredToken\|error occurred"; then
    echo "$(date -u '+%H:%M:%S') Credential error — retrying in 30s"; sleep 30; continue
  fi
  echo "$(date -u '+%H:%M:%S')  EarthCARE=$OUTPUT"
  [[ "$OUTPUT" =~ ^(SUCCEEDED|FAILED)$ ]] && echo "Terminal: $OUTPUT" && break
  sleep 60
done
```

## Modifying the container

```sh
container build --platform linux/amd64 -t join-virtualize:latest \
              /Users/ashiklom/projects/join-project/join-scratch/containers/virtualize/

ECR=445567107118.dkr.ecr.us-west-2.amazonaws.com

aws ecr get-login-password --region us-west-2 | \
  container registry login --username AWS --password-stdin $ECR

container image tag join-virtualize:latest $ECR/join-virtualize:latest

container image push $ECR/join-virtualize:latest
```
