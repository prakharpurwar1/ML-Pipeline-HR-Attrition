"""
Data Download Module
This module handles downloading the HR attrition dataset and uploading to S3.
"""

import os
import pandas as pd
import logging
import argparse
import boto3
from io import StringIO
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def download_data(data_path: str = None) -> pd.DataFrame:
    """
    Download or load the HR attrition dataset.
    
    Args:
        data_path: Path to the CSV file. If None, expects the data in default location.
    
    Returns:
        pd.DataFrame: Raw HR attrition data
    """
    try:
        # Create directories if they don't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Default path if not provided
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "HR_Employee_Attrition.csv")
        
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at: {data_path}")
            raise FileNotFoundError(f"Please place the HR dataset at: {data_path}")
        
        # Load the data
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Basic validation
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Save a copy to processed directory
        output_path = os.path.join(DATA_DIR, "raw_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Raw data saved to: {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        bool: True if validation passes
    """
    try:
        # Check for required columns
        required_columns = ['Attrition', 'Age', 'Department', 'JobRole']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for empty dataframe
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        logger.info("Data validation passed")
        return True
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return False


def upload_to_s3(df: pd.DataFrame, bucket_name: str, project_name: str) -> str:
    """
    Upload HR attrition dataset to S3.
    
    Args:
        df: DataFrame to upload
        bucket_name: S3 bucket name
        project_name: Project name for S3 path organization
    
    Returns:
        str: S3 URI of uploaded file
    """
    logger.info(f"Uploading to S3 bucket: {bucket_name}")
    
    try:
        s3_client = boto3.client('s3')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Upload CSV to S3
        s3_key = f"{project_name}/data/raw/HR_Employee_Attrition_{timestamp}.csv"
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=csv_buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"✓ Data uploaded to {s3_uri}")
        
        # Upload metadata
        metadata = {
            'timestamp': timestamp,
            'dataset': 'HR_Employee_Attrition',
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            's3_uri': s3_uri,
            'shape': df.shape,
            'data_types': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'validation_status': 'passed' if validate_data(df) else 'failed'
        }
        
        metadata_key = f"{project_name}/data/raw/metadata_{timestamp}.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        logger.info(f"✓ Metadata uploaded to s3://{bucket_name}/{metadata_key}")
        
        # Also save the latest version without timestamp
        latest_s3_key = f"{project_name}/data/raw/HR_Employee_Attrition_latest.csv"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=latest_s3_key,
            Body=csv_buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
        logger.info(f"✓ Latest version saved to s3://{bucket_name}/{latest_s3_key}")
        
        return s3_uri
    
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise


def download_from_s3(bucket_name: str, project_name: str, local_path: str = None) -> pd.DataFrame:
    """
    Download HR attrition dataset from S3.
    
    Args:
        bucket_name: S3 bucket name
        project_name: Project name for S3 path organization
        local_path: Optional local path to save the file
    
    Returns:
        pd.DataFrame: Downloaded HR attrition data
    """
    logger.info(f"Downloading latest dataset from S3 bucket: {bucket_name}")
    
    try:
        s3_client = boto3.client('s3')
        
        # Download the latest version
        s3_key = f"{project_name}/data/raw/HR_Employee_Attrition_latest.csv"
        
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_csv(response['Body'])
        
        logger.info(f"✓ Data downloaded from s3://{bucket_name}/{s3_key}")
        logger.info(f"Shape: {df.shape}")
        
        # Optionally save to local path
        if local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_csv(local_path, index=False)
            logger.info(f"✓ Data saved locally to: {local_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        raise


def main():
    """
    Main function for CLI usage with S3 integration.
    """
    parser = argparse.ArgumentParser(description='HR Attrition Data Download and S3 Upload')
    parser.add_argument('--bucket', required=False, help='S3 bucket name')
    parser.add_argument('--project', default='hr-attrition', help='Project name for S3 organization')
    parser.add_argument('--data-path', help='Local path to HR dataset CSV file')
    parser.add_argument('--download-from-s3', action='store_true', 
                       help='Download data from S3 instead of uploading')
    parser.add_argument('--local-save-path', help='Local path to save downloaded data')
    args = parser.parse_args()
    
    try:
        if args.download_from_s3:
            # Download from S3 mode
            if not args.bucket:
                raise ValueError("--bucket is required when downloading from S3")
            
            df = download_from_s3(
                bucket_name=args.bucket,
                project_name=args.project,
                local_path=args.local_save_path or os.path.join(DATA_DIR, "HR_Employee_Attrition.csv")
            )
            
            if validate_data(df):
                logger.info("✓ Data download from S3 and validation successful")
            else:
                logger.error("✗ Data validation failed")
                return
        else:
            # Local load and upload to S3 mode
            df = download_data(data_path=args.data_path)
            
            if validate_data(df):
                logger.info("✓ Data download and validation successful")
                print(f"\nDataset Info:")
                print(f"Shape: {df.shape}")
                print(f"\nFirst few rows:")
                print(df.head())
                
                # Upload to S3 if bucket is provided
                if args.bucket:
                    s3_uri = upload_to_s3(df, args.bucket, args.project)
                    logger.info(f"✓ SUCCESS - Data uploaded to {s3_uri}")
                else:
                    logger.info("ℹ No S3 bucket specified. Skipping S3 upload.")
            else:
                logger.error("✗ Data validation failed")
    
    except Exception as e:
        logger.error(f"✗ Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
