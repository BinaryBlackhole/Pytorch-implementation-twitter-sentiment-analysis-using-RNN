"Author: Sagar Chakraborty"

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence





class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        #forward propagation
        # text = [sent len, batch size]
        # first pass the text into embedding layer
        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        #
        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


