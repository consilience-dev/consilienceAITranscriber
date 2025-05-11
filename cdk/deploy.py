#!/usr/bin/env python3
"""
Deployment script for ConsilienceAITranscriber

This script automates the process of:
1. Building the Docker image
2. Pushing it to Amazon ECR
3. Deploying the CDK stack

It follows an incremental approach, with clear error messages and validation
at each step of the process.
"""
import os
import sys
import argparse
import subprocess
import boto3
import json
from pathlib import Path

# Setup basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("deploy")

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
IMAGE_NAME = "consilience-transcriber"
DEFAULT_TAG = "latest"
DEFAULT_REGION = "us-west-2"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy ConsilienceAITranscriber")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Docker image tag")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build")
    parser.add_argument("--skip-push", action="store_true", help="Skip Docker push")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip CDK deploy")
    return parser.parse_args()


def run_command(cmd, cwd=None):
    """Run a shell command and log output"""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd or PROJECT_ROOT, 
            check=True, 
            text=True, 
            capture_output=True
        )
        logger.info(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)
        sys.exit(1)


def build_docker_image(tag):
    """Build the Docker image locally"""
    logger.info(f"Building Docker image: {IMAGE_NAME}:{tag}")
    run_command(["docker", "build", "-t", f"{IMAGE_NAME}:{tag}", "."])
    logger.info("Docker image built successfully")


def push_to_ecr(region, tag):
    """Push Docker image to Amazon ECR"""
    logger.info(f"Pushing Docker image to ECR in {region}")
    
    # Create ECR client
    ecr = boto3.client('ecr', region_name=region)
    
    # Get AWS account ID
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()["Account"]
    
    # Get repository URI
    repository_name = IMAGE_NAME
    repository_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
    
    try:
        # Check if repository exists, create if it doesn't
        try:
            ecr.describe_repositories(repositoryNames=[repository_name])
            logger.info(f"Repository {repository_name} exists")
        except ecr.exceptions.RepositoryNotFoundException:
            logger.info(f"Creating repository {repository_name}")
            ecr.create_repository(repositoryName=repository_name)
        
        # Get ECR login token
        login_cmd = ecr.get_authorization_token()
        token = login_cmd["authorizationData"][0]["authorizationToken"]
        endpoint = login_cmd["authorizationData"][0]["proxyEndpoint"]
        
        # Log in to ECR
        cmd = ["docker", "login", "--username", "AWS", "--password-stdin", endpoint]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        process.communicate(input=token.encode())
        
        # Tag the image
        run_command(["docker", "tag", f"{IMAGE_NAME}:{tag}", f"{repository_uri}:{tag}"])
        
        # Push the image
        run_command(["docker", "push", f"{repository_uri}:{tag}"])
        logger.info(f"Successfully pushed {repository_uri}:{tag}")
        
        return repository_uri
    except Exception as e:
        logger.error(f"Error pushing to ECR: {str(e)}")
        sys.exit(1)


def deploy_cdk_stack(region):
    """Deploy the CDK stack"""
    logger.info("Deploying CDK stack")
    
    # Ensure AWS_REGION is set for CDK
    os.environ["AWS_REGION"] = region
    
    # Change to CDK directory
    cdk_dir = PROJECT_ROOT / "cdk"
    
    # Install CDK dependencies if needed
    if not (cdk_dir / ".venv").exists():
        logger.info("Setting up CDK virtual environment")
        run_command(["python3", "-m", "venv", ".venv"], cwd=cdk_dir)
        run_command([".venv/bin/pip", "install", "-r", "requirements.txt"], cwd=cdk_dir)
    
    # Run CDK bootstrap if needed
    try:
        run_command([".venv/bin/python", "-m", "aws_cdk.bootstrap", f"--region={region}"], cwd=cdk_dir)
    except Exception as e:
        logger.warning(f"CDK bootstrap issue: {str(e)}")
    
    # Deploy the stack
    run_command([".venv/bin/python", "-m", "aws_cdk", "deploy", "--require-approval=never"], cwd=cdk_dir)
    logger.info("CDK stack deployed successfully")


def main():
    """Main deployment function"""
    args = parse_args()
    
    logger.info("Starting ConsilienceAITranscriber deployment")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Build Docker image
    if not args.skip_build:
        build_docker_image(args.tag)
    else:
        logger.info("Skipping Docker build")
    
    # Push to ECR
    if not args.skip_push:
        repository_uri = push_to_ecr(args.region, args.tag)
    else:
        logger.info("Skipping Docker push")
    
    # Deploy CDK stack
    if not args.skip_deploy:
        deploy_cdk_stack(args.region)
    else:
        logger.info("Skipping CDK deployment")
    
    logger.info("Deployment completed successfully")


if __name__ == "__main__":
    main()
