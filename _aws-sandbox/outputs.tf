output "instance_id" {
  description = "The EC2 instance ID"
  value       = aws_instance.sandbox.id
}

output "instance_state" {
  description = "Current state of the EC2 instance"
  value       = aws_ec2_instance_state.sandbox.state
}

output "instance_type" {
  description = "Instance type"
  value       = aws_instance.sandbox.instance_type
}

output "ami_id" {
  description = "AMI used for the instance"
  value       = aws_instance.sandbox.ami
}

output "availability_zone" {
  description = "Availability zone of the instance"
  value       = aws_instance.sandbox.availability_zone
}

output "subnet_id" {
  description = "Subnet ID of the instance"
  value       = aws_instance.sandbox.subnet_id
}

output "iam_role_name" {
  description = "IAM role attached to the instance"
  value       = aws_iam_role.sandbox.name
}
