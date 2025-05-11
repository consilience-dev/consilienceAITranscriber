"""
SageMaker Stack for ConsilienceAITranscriber

This module defines the AWS CDK stack for deploying the transcription service
to Amazon SageMaker with appropriate IAM roles, ECR repository, and serverless
configuration.
"""
from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
)
from constructs import Construct


class SageMakerTranscriberStack(Stack):
    """
    CDK Stack for ConsilienceAI Transcriber SageMaker deployment.
    
    This stack creates:
    - ECR repository for the Docker image
    - S3 bucket for input/output data
    - IAM roles with appropriate permissions
    - SageMaker model and endpoint configuration
    - SageMaker serverless inference endpoint
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Step 1: Create an ECR repository for the Docker image
        repository = ecr.Repository(
            self, 
            "TranscriberRepository",
            repository_name="consilience-transcriber",
            removal_policy=RemovalPolicy.RETAIN,
            image_scan_on_push=True,
        )

        # Step 2: Create S3 bucket for input/output data
        data_bucket = s3.Bucket(
            self,
            "TranscriberDataBucket",
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
            versioned=True,
        )

        # Step 3: Create IAM role for SageMaker
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess"),
            ],
        )

        # Grant ECR pull permissions to SageMaker role
        repository.grant_pull(sagemaker_role)
        
        # Grant S3 read/write permissions to SageMaker role
        data_bucket.grant_read_write(sagemaker_role)

        # Step 4: Create SageMaker model
        model = sagemaker.CfnModel(
            self,
            "TranscriberModel",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/{repository.repository_name}:latest",
                environment={
                    "WHISPER_MODEL": "base",
                    "MAX_CHUNK_DURATION_MS": "300000",  # 5 minutes
                },
            ),
            model_name="consilience-transcriber-model",
        )

        # Step 5: Create SageMaker Serverless Inference Config
        serverless_config = sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
            max_concurrency=5,
            memory_size_in_mb=6144,  # 6 GB memory
        )

        # Step 6: Create SageMaker Endpoint Config with Serverless Inference
        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "TranscriberEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.attr_model_name,
                    variant_name="AllTraffic",
                    serverless_config=serverless_config,
                )
            ],
            endpoint_config_name="consilience-transcriber-config",
        )

        # Step 7: Create SageMaker Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self,
            "TranscriberEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
            endpoint_name="consilience-transcriber",
        )

        # Step 8: Add SageMaker endpoint URL to the outputs
        CfnOutput(
            self, 
            "EndpointName",
            value=endpoint.endpoint_name or "consilience-transcriber",
            description="Name of the SageMaker endpoint",
        )
        
        CfnOutput(
            self, 
            "DataBucketName",
            value=data_bucket.bucket_name,
            description="S3 bucket for transcriber data",
        )
        
        CfnOutput(
            self, 
            "DockerRepositoryUri", 
            value=repository.repository_uri,
            description="ECR repository URI for the Docker image",
        )
