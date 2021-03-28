import torch
import torch.nn as nn
import torch.nn.functional as F
from data import MAX_LENGTH

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size).to(self.device)
        self.gru = nn.GRU(hidden_size, hidden_size).to(self.device)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size).to(self.device)
        self.gru = nn.GRU(hidden_size, hidden_size).to(self.device)
        self.out = nn.Linear(hidden_size, output_size).to(self.device)
        self.softmax = nn.LogSoftmax(dim=1).to(self.device)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size).to(self.device)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length).to(self.device)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.dropout = nn.Dropout(self.dropout_p).to(self.device)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size).to(self.device)
        self.out = nn.Linear(self.hidden_size, self.output_size).to(self.device)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_out = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        attn_weights = F.softmax(attn_out, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
