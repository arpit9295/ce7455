# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--seq_size', type=int, default=35,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx_export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--dry_run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--skip_train', action='store_true',
                    help='skip training')

args = parser.parse_args()

print()
print("Configuration", args.__dict__)
print()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).contiguous()
    return data.to(device)

train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.eval_batch_size)
test_data = batchify(corpus.test, args.eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.FNNModel(args.seq_size, ntokens, args.emsize, args.nhid, args.tied).to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), args.lr)

###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.seq_size.
# If source is equal to the example output of the batchify function, with
# a seq_size-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    if (i + args.seq_size + 1 >= source.size(1)):
        raise Exception('batch index exceeding')
    seq_len = args.seq_size
    source_t = source.t()
    data = source_t[i:i+seq_len].t()
    target = source_t[i+seq_len:i+seq_len+1].t().view(-1)
    return data, target

def evaluate(data_source):
    total_loss = 0.

    batches_per_iteration = data_source.size(0)
    num_iterations = data_source.size(1) - args.seq_size - 1

    with torch.no_grad():
        # check on dimensions of data_source
        for i in range(0, num_iterations):
            data, targets = get_batch(data_source, i)
            output = model(data)
            total_loss += batches_per_iteration * criterion(output, targets).item()
    return total_loss / (batches_per_iteration * num_iterations)

def train():
    total_loss = 0.
    start_time = time.time()
    for i in range(0, train_data.size(1) - args.seq_size - 1):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, train_data.size(1) - args.seq_size, args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

# Loop over epochs.
best_val_loss = None

if not args.skip_train:
  # At any point you can hit Ctrl + C to break out of training early.
  try:
      for epoch in range(1, args.epochs+1):
          epoch_start_time = time.time()
          train()
          val_loss = evaluate(val_data)
          print('-' * 89)
          print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
          print('-' * 89)
          # Save the model if the validation loss is the best we've seen so far.
          if not best_val_loss or val_loss < best_val_loss:
              with open(args.save, 'wb') as f:
                  torch.save(model, f)
              best_val_loss = val_loss
          # else:
          #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
          #     optimizer = torch.optim.Adam(model.parameters(), lr)

  except KeyboardInterrupt:
      print('-' * 89)
      print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.seq_size)
