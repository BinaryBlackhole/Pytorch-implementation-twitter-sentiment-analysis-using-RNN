"""
Author: Sagar Chakraborty
Updated: Modern PyTorch 2.x implementation using DataLoader instead of legacy torchtext
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import SentimentLSTM, SentimentRNN
from text_cleaning import cleanup_text, simple_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Vocabulary:
    """Build and manage vocabulary for text data."""

    def __init__(self, max_size: int = 25000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_freq = {}

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts."""
        logger.info("Building vocabulary...")

        # Count word frequencies
        for text in texts:
            tokens = simple_tokenizer(text)
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1

        # Sort by frequency and filter by min_freq
        sorted_words = sorted(
            [(word, freq) for word, freq in self.word_freq.items() if freq >= self.min_freq],
            key=lambda x: x[1],
            reverse=True
        )

        # Add words up to max_size
        for word, freq in sorted_words[:self.max_size - 2]:  # -2 for <pad> and <unk>
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(f"Vocabulary built with {len(self.word2idx)} tokens")

    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices."""
        tokens = simple_tokenizer(text)
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis."""

    def __init__(self, data_df: pd.DataFrame, vocab: Vocabulary, max_length: int = 200):
        self.data = data_df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length

        # Clean texts
        logger.info("Cleaning texts...")
        self.cleaned_texts = cleanup_text(self.data['sentences'].tolist())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.cleaned_texts[idx]
        label = float(self.data.iloc[idx]['labels'])

        # Encode text
        encoded = self.vocab.encode(text)

        # Truncate or pad
        actual_length = min(len(encoded), self.max_length)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [0] * (self.max_length - len(encoded))

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
            actual_length
        )


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader."""
    texts, labels, lengths = zip(*batch)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return texts, labels, lengths


class SentimentTrainer:
    """Trainer class for sentiment analysis model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_dir: str,
        model_name: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_name = model_name

        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def binary_accuracy(preds: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate binary accuracy."""
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc.item()

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0

        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)

        for texts, labels, lengths in progress_bar:
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(texts, lengths).squeeze(1)
            loss = self.criterion(predictions, labels)
            acc = self.binary_accuracy(predictions, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'acc': acc})

        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0

        with torch.no_grad():
            for texts, labels, lengths in data_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                predictions = self.model(texts, lengths).squeeze(1)
                loss = self.criterion(predictions, labels)
                acc = self.binary_accuracy(predictions, labels)

                epoch_loss += loss.item()
                epoch_acc += acc

        return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

    def train(self, num_epochs: int) -> None:
        """Train the model for specified number of epochs."""
        best_valid_loss = float('inf')

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.evaluate(self.valid_loader)

            end_time = time.time()
            epoch_mins = int((end_time - start_time) / 60)
            epoch_secs = int((end_time - start_time) % 60)

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_path = self.model_dir / self.model_name
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")

            logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            logger.info(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')

    def test(self) -> Tuple[float, float]:
        """Test the model on test set."""
        # Load best model
        model_path = self.model_dir / self.model_name
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Loaded model from {model_path}")

        test_loss, test_acc = self.evaluate(self.test_loader)
        logger.info(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

        return test_loss, test_acc


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = 'project_config.json') -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main training pipeline."""
    # Set seed for reproducibility
    set_seed(42)

    # Load configuration
    config = load_config('project_config.json')

    # Extract hyperparameters
    DATA_PATH = config['data_path']
    MODEL_DIR = config['model_dir']
    MODEL_NAME = config['model_name']
    EMBEDDING_DIM = int(config['embedding_dim'])
    HIDDEN_DIM = int(config['hidden_dim'])
    OUTPUT_DIM = int(config['output_dim'])
    BATCH_SIZE = int(config['batch_size'])
    MAX_VOCAB_SIZE = int(config['max_vocab_size'])
    LEARNING_RATE = float(config['learning_rate'])
    NUM_EPOCHS = int(config['epoch'])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Build vocabulary from training data
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=1)
    train_texts = cleanup_text(train_df['sentences'].tolist())
    vocab.build_vocab(train_texts)

    # Create datasets
    full_train_dataset = SentimentDataset(train_df, vocab)
    test_dataset = SentimentDataset(test_df, vocab)

    # Split training data into train and validation
    train_size = int(0.7 * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    train_dataset, valid_dataset = random_split(
        full_train_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(valid_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Initialize model (use modern LSTM by default)
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_idx=vocab.word2idx['<pad>']
    )

    logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Create trainer
    trainer = SentimentTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_dir=MODEL_DIR,
        model_name=MODEL_NAME
    )

    # Train model
    trainer.train(num_epochs=NUM_EPOCHS)

    # Test model
    trainer.test()


if __name__ == '__main__':
    main()
