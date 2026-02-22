"""
Data Preprocessing Script for HR Attrition Dataset
Runs on SageMaker Processing Job

This script:
1. Loads raw HR data from S3
2. Handles missing values
3. Engineers HR-specific features
4. Splits and normalizes data
5. Saves processed data back to S3
"""

import os
import sys
import pandas as pd
import numpy as np
import boto3
import pickle
import argparse
import logging
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_latest_data_from_s3(bucket_name, data_prefix):
    """Load the most recent raw HR attrition data from S3"""
    logger.info(f"Loading data from s3://{bucket_name}/{data_prefix}/data/raw/")

    s3_client = boto3.client('s3')

    # List all CSV files in the raw data folder
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=f"{data_prefix}/data/raw/"
    )

    if 'Contents' not in response:
        raise FileNotFoundError(f"No files found in s3://{bucket_name}/{data_prefix}/data/raw/")

    # Find the most recent CSV file (prioritize latest, then timestamped files)
    csv_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found")

    # Prefer _latest.csv file, otherwise use most recent
    latest_csv = next((f for f in csv_files if 'latest' in f['Key']), None)
    if latest_csv:
        s3_key = latest_csv['Key']
    else:
        latest_file = max(csv_files, key=lambda x: x['LastModified'])
        s3_key = latest_file['Key']

    logger.info(f"Loading file: {s3_key}")

    # Download and read CSV
    obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

    logger.info(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")

    return df, s3_key


def handle_missing_values(df):
    """Handle missing values in the HR dataset"""
    logger.info("Handling missing values...")

    missing_before = df.isnull().sum().sum()
    logger.info(f"  Missing values before: {missing_before}")

    if missing_before > 0:
        logger.info(f"  Missing by column:")
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                logger.info(f"    {col}: {missing}")

    # Numerical features: fill with median
    numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
                         'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                         'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                         'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                         'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                         'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                         'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    for col in numerical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  ✓ Filled {col} with median: {median_val}")

    # Categorical features: fill with mode
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 
                           'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    
    for col in categorical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"  ✓ Filled {col} with mode: {mode_val}")

    missing_after = df.isnull().sum().sum()
    logger.info(f"✓ Missing values after: {missing_after}")

    return df


def safe_cut(series, bins, labels):
    """Safely bin a continuous variable into categories"""
    try:
        return pd.cut(series, bins=bins, labels=labels)
    except Exception:
        return pd.qcut(series.fillna(series.median()), q=len(labels), 
                      labels=labels, duplicates='drop')


def engineer_features(df):
    """Create HR-specific features from existing ones"""
    logger.info("Engineering HR-specific features...")

    # Binary target (Attrition: Yes=1, No=0)
    if 'Attrition' in df.columns:
        df['AttritionFlag'] = df['Attrition'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
        # Handle any numeric values already present
        if df['AttritionFlag'].isnull().any():
            df['AttritionFlag'] = pd.to_numeric(df['Attrition'], errors='coerce').fillna(df['AttritionFlag'])
        
        attrition_count = (df['AttritionFlag'] == 1).sum()
        no_attrition_count = (df['AttritionFlag'] == 0).sum()
        logger.info(f"  Created AttritionFlag:")
        logger.info(f"    No Attrition (0): {no_attrition_count}")
        logger.info(f"    Attrition (1): {attrition_count}")

    # Age groups
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['AgeGroup'] = safe_cut(df['Age'], bins=[18, 30, 40, 50, 60, 100],
                                  labels=['18-30', '31-40', '41-50', '51-60', '60+'])
        logger.info(f"  ✓ Created AgeGroup feature")

    # Tenure buckets (early stage attrition)
    if 'YearsAtCompany' in df.columns:
        df['YearsAtCompany_filled'] = df['YearsAtCompany'].fillna(0)
        df['TenureBucket'] = pd.cut(df['YearsAtCompany_filled'], bins=[-0.1, 2, 5, 10, 100],
                                    labels=['<2', '2-5', '6-10', '10+'])
        logger.info(f"  ✓ Created TenureBucket feature")

    # Interaction: JobSatisfaction x OverTime
    if 'JobSatisfaction' in df.columns and 'OverTime' in df.columns:
        df['JobSatisfaction_filled'] = df['JobSatisfaction'].fillna(df['JobSatisfaction'].median())
        df['SatisfactionOverTime'] = (
            df['JobSatisfaction_filled'].astype(int).astype(str) + '_' + df['OverTime'].astype(str)
        )
        logger.info(f"  ✓ Created SatisfactionOverTime feature")

    # Percent of years in current role (relative to company tenure)
    if 'YearsInCurrentRole' in df.columns and 'YearsAtCompany' in df.columns:
        df['YearsAtCompany_safe'] = df['YearsAtCompany'].fillna(0) + 1
        df['YearsInCurrentRole_Percent'] = df['YearsInCurrentRole'].fillna(0) / df['YearsAtCompany_safe']
        logger.info(f"  ✓ Created YearsInCurrentRole_Percent feature")

    # High tenure satisfied (retention indicator)
    if 'TenureBucket' in df.columns and 'JobSatisfaction' in df.columns:
        df['HighTenure_Satisfied'] = (
            (df['TenureBucket'] == '10+') & 
            (df['JobSatisfaction'].fillna(0) >= 3)
        ).astype(int)
        high_tenure_satisfied = df['HighTenure_Satisfied'].sum()
        logger.info(f"  ✓ Created HighTenure_Satisfied: {high_tenure_satisfied} employees")

    # Income-to-experience ratio
    if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
        df['TotalWorkingYears_safe'] = df['TotalWorkingYears'].fillna(1) + 1
        df['Income_Experience_Ratio'] = df['MonthlyIncome'] / df['TotalWorkingYears_safe']
        logger.info(f"  ✓ Created Income_Experience_Ratio feature")

    # Work-life balance flags
    if 'WorkLifeBalance' in df.columns:
        df['Poor_WorkLife'] = (df['WorkLifeBalance'] <= 2).astype(int)
        poor_count = df['Poor_WorkLife'].sum()
        logger.info(f"  ✓ Created Poor_WorkLife flag: {poor_count} employees")

    # Distance from home flag
    if 'DistanceFromHome' in df.columns:
        df['Long_Commute'] = (df['DistanceFromHome'] > 20).astype(int)
        long_commute_count = df['Long_Commute'].sum()
        logger.info(f"  ✓ Created Long_Commute flag: {long_commute_count} employees")

    # Low salary hike flag
    if 'PercentSalaryHike' in df.columns:
        df['Low_Salary_Hike'] = (df['PercentSalaryHike'] <= 12).astype(int)
        low_hike_count = df['Low_Salary_Hike'].sum()
        logger.info(f"  ✓ Created Low_Salary_Hike flag: {low_hike_count} employees")

    # Drop temporary helper columns
    temp_cols = ['YearsAtCompany_filled', 'JobSatisfaction_filled', 'YearsAtCompany_safe', 'TotalWorkingYears_safe']
    for c in temp_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    logger.info(f"✓ Feature engineering complete (Total features: {len(df.columns)})")

    return df


def prepare_features_for_modeling(df):
    """Prepare features by encoding categorical variables"""
    logger.info("Preparing features for modeling...")
    
    # Identify categorical columns to encode
    cat_candidates = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                     'JobRole', 'MaritalStatus', 'OverTime', 'Over18']
    cat_cols = [c for c in cat_candidates if c in df.columns]
    
    # Add engineered categorical features
    for c in ['AgeGroup', 'TenureBucket', 'SatisfactionOverTime']:
        if c in df.columns:
            cat_cols.append(c)
    
    # Convert to category type
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    logger.info(f"  ✓ Encoded {len(cat_cols)} categorical features")
    
    # Get all feature columns (exclude target and identifiers)
    exclude_cols = ['Attrition', 'AttritionFlag', 'EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]
    
    # Ensure all features are numeric
    X = df_encoded[feature_cols].select_dtypes(include=[np.number]).values
    y = df_encoded['AttritionFlag'].fillna(0).astype(int).values
    
    logger.info(f"  ✓ Feature matrix prepared: {X.shape}")
    logger.info(f"  ✓ Target vector prepared: {y.shape}")
    
    return X, y, feature_cols


def split_and_normalize(df):
    """Split data into train/val/test and normalize"""
    logger.info("Splitting and normalizing data...")

    # Prepare features
    X, y, feature_cols = prepare_features_for_modeling(df)
    
    logger.info(f"  Feature matrix: {X.shape}")
    logger.info(f"  Target vector: {y.shape}")
    logger.info(f"  Number of features: {len(feature_cols)}")

    # Check class balance
    class_counts = np.bincount(y)
    logger.info(f"  Class distribution:")
    logger.info(f"    No Attrition (0): {class_counts[0]}")
    logger.info(f"    Attrition (1): {class_counts[1]}")

    # Split: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    logger.info(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
    logger.info(f"  Val set: {X_val.shape[0]} samples ({X_val.shape[0] / len(X) * 100:.1f}%)")
    logger.info(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

    # Class distribution in splits
    logger.info(f"  Class distribution in splits:")
    logger.info(f"    Train - Attrition: {y_train.sum()}, No Attrition: {len(y_train) - y_train.sum()}")
    logger.info(f"    Val - Attrition: {y_val.sum()}, No Attrition: {len(y_val) - y_val.sum()}")
    logger.info(f"    Test - Attrition: {y_test.sum()}, No Attrition: {len(y_test) - y_test.sum()}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"✓ Normalization complete")
    logger.info(f"  Mean: {X_train_scaled.mean():.6f} (should be ~0)")
    logger.info(f"  Std: {X_train_scaled.std():.6f} (should be ~1)")

    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler
    }


def save_to_s3(data, bucket_name, data_prefix, output_path=None):
    """Save processed data to S3"""
    logger.info("Saving processed data to S3...")

    s3_client = boto3.client('s3')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine save path (SageMaker or direct)
    if output_path:
        # Running in SageMaker Processing - save to local output first
        base_path = output_path
        logger.info(f"  Using SageMaker output path: {base_path}")
    else:
        # Running standalone - save directly to S3
        base_path = '/tmp'
        logger.info(f"  Using temp path: {base_path}")

    # Save numpy arrays
    npz_filename = f'hr_attrition_processed_{timestamp}.npz'
    npz_path = os.path.join(base_path, npz_filename)

    np.savez(
        npz_path,
        X_train=data['X_train'],
        X_val=data['X_val'],
        X_test=data['X_test'],
        y_train=data['y_train'],
        y_val=data['y_val'],
        y_test=data['y_test']
    )
    logger.info(f"  ✓ Saved NPZ to {npz_path}")

    # Save scaler
    scaler_filename = f'scaler_{timestamp}.pkl'
    scaler_path = os.path.join(base_path, scaler_filename)
    with open(scaler_path, 'wb') as f:
        pickle.dump(data['scaler'], f)
    logger.info(f"  ✓ Saved scaler to {scaler_path}")

    # Save feature names
    features_filename = f'feature_names_{timestamp}.json'
    features_path = os.path.join(base_path, features_filename)
    with open(features_path, 'w') as f:
        json.dump(data['feature_names'], f, indent=2)
    logger.info(f"  ✓ Saved feature names to {features_path}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'dataset': 'HR_Employee_Attrition',
        'train_samples': int(len(data['y_train'])),
        'val_samples': int(len(data['y_val'])),
        'test_samples': int(len(data['y_test'])),
        'num_features': int(data['X_train'].shape[1]),
        'feature_names': data['feature_names'],
        'train_attrition_count': int(data['y_train'].sum()),
        'val_attrition_count': int(data['y_val'].sum()),
        'test_attrition_count': int(data['y_test'].sum()),
        'train_retention_count': int(len(data['y_train']) - data['y_train'].sum()),
        'val_retention_count': int(len(data['y_val']) - data['y_val'].sum()),
        'test_retention_count': int(len(data['y_test']) - data['y_test'].sum())
    }

    metadata_filename = f'preprocessing_metadata_{timestamp}.json'
    metadata_path = os.path.join(base_path, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata to {metadata_path}")

    # If not in SageMaker, upload to S3 manually
    if not output_path:
        # Upload NPZ
        npz_key = f"{data_prefix}/data/processed/{npz_filename}"
        s3_client.upload_file(npz_path, bucket_name, npz_key)
        logger.info(f"  ✓ Uploaded to s3://{bucket_name}/{npz_key}")

        # Upload scaler
        scaler_key = f"{data_prefix}/models/scalers/{scaler_filename}"
        s3_client.upload_file(scaler_path, bucket_name, scaler_key)
        logger.info(f"  ✓ Uploaded to s3://{bucket_name}/{scaler_key}")

        # Upload feature names
        features_key = f"{data_prefix}/data/processed/{features_filename}"
        s3_client.upload_file(features_path, bucket_name, features_key)
        logger.info(f"  ✓ Uploaded to s3://{bucket_name}/{features_key}")

        # Upload metadata
        metadata_key = f"{data_prefix}/data/processed/{metadata_filename}"
        s3_client.upload_file(metadata_path, bucket_name, metadata_key)
        logger.info(f"  ✓ Uploaded to s3://{bucket_name}/{metadata_key}")

        metadata['npz_s3_uri'] = f"s3://{bucket_name}/{npz_key}"
        metadata['scaler_s3_uri'] = f"s3://{bucket_name}/{scaler_key}"

    logger.info("✓ All files saved successfully")

    return timestamp, metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess HR attrition data')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='hr-attrition', help='Data prefix in S3')
    parser.add_argument('--output-path', default=None, help='Output path (for SageMaker)')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("HR ATTRITION DATA PREPROCESSING")
    logger.info("=" * 70)
    logger.info(f"  Bucket: {args.bucket}")
    logger.info(f"  Prefix: {args.prefix}")
    logger.info(f"  Output: {args.output_path or 'Direct to S3'}")
    logger.info("=" * 70)

    try:
        # Step 1: Load data
        df, source_file = load_latest_data_from_s3(args.bucket, args.prefix)

        # Step 2: Handle missing values
        df = handle_missing_values(df)

        # Step 3: Engineer features
        df = engineer_features(df)

        # Step 4: Split and normalize
        processed_data = split_and_normalize(df)

        # Step 5: Save to S3
        timestamp, metadata = save_to_s3(
            processed_data,
            args.bucket,
            args.prefix,
            args.output_path
        )

        logger.info("\n" + "=" * 70)
        logger.info("✓ PREPROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Source: {source_file}")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Train: {metadata['train_samples']} samples")
        logger.info(f"  Val: {metadata['val_samples']} samples")
        logger.info(f"  Test: {metadata['test_samples']} samples")
        logger.info(f"  Features: {metadata['num_features']}")
        logger.info(f"  Attrition Rate:")
        logger.info(f"    Train: {metadata['train_attrition_count']}/{metadata['train_samples']} ({metadata['train_attrition_count']/metadata['train_samples']*100:.1f}%)")
        logger.info(f"    Val: {metadata['val_attrition_count']}/{metadata['val_samples']} ({metadata['val_attrition_count']/metadata['val_samples']*100:.1f}%)")
        logger.info(f"    Test: {metadata['test_attrition_count']}/{metadata['test_samples']} ({metadata['test_attrition_count']/metadata['test_samples']*100:.1f}%)")
        logger.info("=" * 70)

        # Write success marker for SageMaker
        if args.output_path:
            with open(os.path.join(args.output_path, 'SUCCESS'), 'w') as f:
                f.write('HR Attrition preprocessing completed successfully\n')

        return 0

    except Exception as e:
        logger.error(f"✗ PREPROCESSING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
