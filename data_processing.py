"""
Author: Sagar Chakraborty
Updated: Modern implementation with pathlib, type hints, and better practices
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'data_config.json') -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_sentiment_dataset(
    dataset_path: str,
    num_neg_labels: int,
    num_pos_labels: int,
    train_filepath: str,
    test_filepath: str,
    num_training_samples: int
) -> None:
    """
    Process sentiment140 dataset and split into train/test sets.

    Args:
        dataset_path: Path to main dataset CSV
        num_neg_labels: Number of negative samples to extract
        num_pos_labels: Number of positive samples to extract
        train_filepath: Output path for training data
        test_filepath: Output path for test data
        num_training_samples: Number of samples for training set
    """
    logger.info(f"Loading dataset from {dataset_path}")

    # Load the full dataset
    dataset_df = pd.read_csv(
        dataset_path,
        names=['labels', 'id', 'datetime', 'query', 'username', 'sentences'],
        header=None,
        sep=',',
        encoding="ISO-8859-1"
    )

    logger.info(f"Dataset loaded with {len(dataset_df)} samples")
    logger.info(f"Label distribution:\n{dataset_df['labels'].value_counts()}")

    # Sample from each class
    logger.info(f"Sampling {num_neg_labels} negative and {num_pos_labels} positive samples")
    filter_0_df = dataset_df[dataset_df['labels'] == 0].sample(
        n=num_neg_labels,
        random_state=42
    )
    filter_4_df = dataset_df[dataset_df['labels'] == 4].sample(
        n=num_pos_labels,
        random_state=42
    )

    # Drop unnecessary columns
    columns_to_drop = ['id', 'datetime', 'query', 'username']
    filter_0_df = filter_0_df.drop(columns=columns_to_drop)
    filter_4_df = filter_4_df.drop(columns=columns_to_drop)

    logger.info(f"Negative samples: {len(filter_0_df)}, Positive samples: {len(filter_4_df)}")

    # Merge dataframes
    final_df = pd.concat([filter_0_df, filter_4_df], ignore_index=True)

    # Remove rows with URLs (likely spam)
    logger.info("Removing tweets containing URLs")
    initial_count = len(final_df)
    final_df = final_df[~final_df['sentences'].str.contains(
        r'http\S+|www\.\S+',
        na=False,
        regex=True
    )]
    logger.info(f"Removed {initial_count - len(final_df)} tweets with URLs")

    # Shuffle the dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert labels to binary (0 and 1)
    # Replace label 4 with 1
    final_df['labels'] = final_df['labels'].replace(4, 1)

    logger.info(f"Final dataset size: {len(final_df)}")
    logger.info(f"Binary label distribution:\n{final_df['labels'].value_counts()}")

    # Split into train and test
    train_df = final_df[:num_training_samples]
    test_df = final_df[num_training_samples:]

    logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Create output directories if they don't exist
    train_path = Path(train_filepath)
    test_path = Path(test_filepath)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV without index
    logger.info(f"Saving training data to {train_filepath}")
    train_df.to_csv(train_filepath, index=False)

    logger.info(f"Saving test data to {test_filepath}")
    test_df.to_csv(test_filepath, index=False)

    logger.info("Data processing completed successfully!")


def main():
    """Main function to execute data processing pipeline."""
    # Load configuration
    config_data = load_config('data_config.json')

    # Extract configuration parameters
    dataset_path = config_data['dataset_full_path']
    num_neg_labels = int(config_data['num_neg_labels'])
    num_pos_labels = int(config_data['num_pos_labels'])
    train_filepath = config_data['trainset_fullpath']
    test_filepath = config_data['testset_fullpath']
    num_training_samples = int(config_data['num_training_sample'])

    # Process the dataset
    process_sentiment_dataset(
        dataset_path=dataset_path,
        num_neg_labels=num_neg_labels,
        num_pos_labels=num_pos_labels,
        train_filepath=train_filepath,
        test_filepath=test_filepath,
        num_training_samples=num_training_samples
    )


if __name__ == '__main__':
    main()
