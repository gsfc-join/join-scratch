tf_dir := "_aws-sandbox"
tfout   := "_aws-sandbox/_tfout.json"

# Create/update the sandbox instance (terraform apply, then save outputs)
sandbox_create:
    cd {{tf_dir}} && terraform apply
    cd {{tf_dir}} && terraform output -json > _tfout.json

# Open an interactive SSM session on the sandbox instance
sandbox_connect:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(jq -r '.instance_id.value' {{tfout}})
    REGION=$(cd {{tf_dir}} && terraform output -json 2>/dev/null | jq -r '.aws_region.value // "us-west-2"')
    aws ssm start-session --target "$INSTANCE_ID" --region "${REGION:-us-west-2}"

# Check the current status of the sandbox instance
sandbox_status:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(jq -r '.instance_id.value' {{tfout}})
    aws ec2 describe-instance-status \
        --instance-ids "$INSTANCE_ID" \
        --region us-west-2 \
        --include-all-instances \
        --query 'InstanceStatuses[0].InstanceState.Name' \
        --output text

# Start the sandbox instance if it isn't already running, then wait until ready
sandbox_start:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(jq -r '.instance_id.value' {{tfout}})
    STATE=$(aws ec2 describe-instance-status \
        --instance-ids "$INSTANCE_ID" \
        --region us-west-2 \
        --include-all-instances \
        --query 'InstanceStatuses[0].InstanceState.Name' \
        --output text)
    if [ "$STATE" = "running" ]; then
        echo "Instance is already running."
    else
        echo "Starting instance $INSTANCE_ID..."
        aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region us-west-2
        echo "Waiting for instance to be ready..."
        aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID" --region us-west-2
        echo "Instance is ready."
    fi

# Stop the sandbox instance if it isn't already stopped
sandbox_stop:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(jq -r '.instance_id.value' {{tfout}})
    STATE=$(aws ec2 describe-instance-status \
        --instance-ids "$INSTANCE_ID" \
        --region us-west-2 \
        --include-all-instances \
        --query 'InstanceStatuses[0].InstanceState.Name' \
        --output text)
    if [ "$STATE" = "stopped" ]; then
        echo "Instance is already stopped."
    else
        echo "Stopping instance $INSTANCE_ID..."
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region us-west-2
        echo "Waiting for instance to stop..."
        aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region us-west-2
        echo "Instance stopped."
    fi

# Destroy the sandbox infrastructure (terraform destroy)
sandbox_destroy:
    cd {{tf_dir}} && terraform destroy
