#!/usr/bin/env python3
"""
Main CDK application entry point for ConsilienceAITranscriber deployment.

This script initializes the AWS CDK app and instantiates the SageMaker stack
for deploying the transcription service.
"""
import os
from aws_cdk import App

from stacks.sagemaker_stack import SageMakerTranscriberStack

app = App()

# Create the main SageMaker transcriber stack
SageMakerTranscriberStack(
    app, 
    "ConsilienceTranscriberStack",
    env={
        "account": os.environ.get("CDK_DEFAULT_ACCOUNT"),
        "region": os.environ.get("CDK_DEFAULT_REGION", "us-west-2")
    },
    description="Consilience AI Transcriber SageMaker deployment"
)

app.synth()
