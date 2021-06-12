"Author: Sagar Chakraborty"
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from text_cleaning import cleanup_text
from model import RNN

from pydoc import locate
from torch.nn.parallel import DataParallel

from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Precision, Recall, Loss

import json
import torch.optim as optim
import time
import os
from misc import create_supervised_evaluator
from misc import ModelLoader
from ignite.engine import Events
import random

from model import RNN



class trainer(object):
    def __init__(self,data_path,model_dir,model_name,device=-1):
        self.data_path= data_path
        self.model_dir = model_dir
        self.model_name= model_name
        self.device = device

    @staticmethod
    def train(model, iterator, optimizer, criterion):
        """Train function to start the training the declared  model. model.train() initialize it.
        for every batch picked up from the iterator we send it to the model and get predictions.
        loss = Predicted_y - Actual_y and based of the loss we calculate accuracy.
        loss.backward is for back propagation"""
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:
            optimizer.zero_grad()

            predictions = model(batch.sentences[0]).squeeze(1)

            loss = criterion(predictions, batch.labels)

            acc = trainer.binary_accuracy(predictions, batch.labels)

            loss.backward() #back propagation

            optimizer.step() #weight update

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    @staticmethod
    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.sentences[0]).squeeze(1)

                loss = criterion(predictions, batch.labels)

                acc = trainer.binary_accuracy(predictions, batch.labels)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs





#This is how the config.json data looks like
"""
{
	"data_path": "data",
	"model_dir": "saved_model",
	"device": "-1",
	"model_name": "sentiment_classifer_rnn_sagar.pt",
	"embedding_dim": "100",
	"hidden_dim": "256",
	"output_dim": "1",
	"batch_size":"64"
	"max_vocab_size": "25000"
}
"""



f = open('project_config.json','r')

config_data = json.loads(f.read())

EMBEDDING_DIM = int(config_data['embedding_dim'])
HIDDEN_DIM = int(config_data['hidden_dim'])
OUTPUT_DIM = int(config_data['output_dim'])
BATCH_SIZE = int(config_data['batch_size'])
MAX_VOCAB_SIZE = int(config_data['max_vocab_size'])

#Parameters we have provided for our model
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256
# OUTPUT_DIM = 1
# BATCH_SIZE = 64
# MAX_VOCAB_SIZE = 25_000


data_path = config_data['data_path']
model_dir = config_data['model_dir']
device = -1
model_name = config_data['model_name']
learning_rate = float(config_data['learning_rate'])
num_epoch = int(config_data['epoch'])



################-------##################
Model_trainer = trainer(data_path,model_dir,model_name,device)

# seed
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    device = None

tokenize = lambda s: s.split()

text = data.Field(
    preprocessing=cleanup_text, include_lengths=True, tokenize=tokenize
)

sentiment = data.LabelField(dtype=torch.float)
train, test = data.TabularDataset.splits(
    Model_trainer.data_path,
    train="train.csv",
    validation="test.csv",
    format="csv",
    fields=[("labels", sentiment), ("sentences", text)],
)

text.build_vocab(train.text, min_freq=1, max_size=MAX_VOCAB_SIZE)
sentiment.build_vocab(train.sentiment)

print(len(train), len(test))

print(vars(train.examples[5]))

train_data, valid_data = train.split(random_state=random.seed(42))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test)}')



text.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
sentiment.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(text.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(sentiment.vocab)}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    datasets=[train_data, valid_data, test],
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.sentences),
    device=device, )

INPUT_DIM = len(text.vocab)
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
print(f'The model has {Model_trainer.count_parameters(model):,} trainable parameters')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)


N_EPOCHS = num_epoch

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc =  Model_trainer.train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = Model_trainer.evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = Model_trainer.epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(Model_trainer.model_dir,Model_trainer.model_name))

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
    print(f'\t Validation Loss: {valid_loss:.3f} |  Validation Accuracy: {valid_acc * 100:.2f}%')


#Loading model from directory and testing : test score and accuracy
model.load_state_dict(torch.load(os.path.join(Model_trainer.model_dir,Model_trainer.model_name)))

test_loss, test_acc = Model_trainer.evaluate(model, test_iterator, criterion)

print(f'Overall Test Loss: {test_loss:.3f} | Overall Test Accuracy: {test_acc * 100:.2f}%')