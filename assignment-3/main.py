from __future__ import print_function
import argparse
import os
import random

import torch
import torch.nn as nn
from torch import optim
import time
import _pickle as cPickle

import sys

from data import SOS_token, EOS_token, MAX_LENGTH,\
                 WordIndexer, CharIndexer,\
                 prepareData
from utils import Unbuffered,\
                  timeSince, savePlot, saveAttention
from model import EncoderRNN, CharEncoderRNN, AttnDecoderRNN, DecoderRNN


# ##### Define constants and paramaters

parser = argparse.ArgumentParser()
# parameters for the Model
parser.add_argument('--data', default="./data", help="Data directory")
parser.add_argument('--lang1', default="eng", help="Code of input lang")
parser.add_argument('--lang2', default="fra", help="Code of output lang")
parser.add_argument('--mapping-file', default="mapping.pkl", help="Path to load/save mapping file")

parser.add_argument('--output-dir', default='logs')
parser.add_argument('--train', action='store_true', help="Training the model rather than loading from file")
parser.add_argument('--use-attention', default='true', help="Use attention, use --attention=false to disable")
parser.add_argument('--encoder-file', default='encoder.model', help="Filename of encoder model")
parser.add_argument('--decoder-file', default='decoder.model', help="Filename of decoder model")

parser.add_argument('--use-char-embedding', action='store_true', help="Hidden layer size")
parser.add_argument('--hidden-size', type=int, default=256, help="Hidden layer size")
parser.add_argument('--char-embedding-size', type=int, default=32, help="Char embedding dimension")
parser.add_argument('--char-representation-size', type=int, default=32, help="Token embedding dimension")
parser.add_argument('--embedding-path', default="./data/glove.6B.100d.txt", help="Location of pretrained embeddings")

parser.add_argument('--train-iters', type=int, default=75000, help="Number of iterations to train for")
parser.add_argument('--learning-rate', type=int, default=0.01)
parser.add_argument('--dropout', type=int, default=0.5, help="Droupout on the input (0 = no dropout)")
parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5, help="Probability of using Teacher Forcing")
parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to run")
parser.add_argument('--weights', default="", help="path to Pretrained for from a previous run")
parser.add_argument('--gradient-clip', default=5.0)
parser.add_argument('--use-gpu', default='true')
parser.add_argument('--plot-every', type=int, default=100)
parser.add_argument('--plot-file', default="losses.png")
parser.add_argument('--print-every', type=int, default=1000)

parameters = vars(parser.parse_args())

output_dir = parameters['output_dir']
use_char_embedding = parameters['use_char_embedding']

if not os.path.exists(output_dir):
    os.system('mkdir ' + output_dir)

f = open(output_dir + '/out.txt', 'w')
sys.stdout = Unbuffered(f) # Change the standard output to the file we created.

use_attention = parameters['use_attention'] != 'false'
use_gpu = parameters['use_gpu'] != 'false'

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, so you should probably run without --use-gpu=false")

device = torch.device("cuda" if use_gpu else "cpu")
mapping_path = output_dir + '/' + parameters['mapping_file']


# if os.path.isfile(mapping_path):
#     with open(mapping_path, 'rb') as f:
#         mappings = cPickle.load(f)
#         input_lang = mappings['input_lang']
#         output_lang = mappings['output_lang']
#         pairs = mappings['pairs']
# else:
#     input_lang, output_lang, pairs = prepareData(parameters['lang1'], parameters['lang2'], True)
#     with open(mapping_path, 'wb') as f:
#         mappings = {
#             input_lang,
#             output_lang,
#             pairs,
#         }
#         cPickle.dump(mappings, f)

input_lang, output_lang, pairs = prepareData(parameters['lang1'], parameters['lang2'], True)

def train(input_word_tensor,
          target_word_tensor,
          input_char_tensor,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion,
          max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_word_tensor.size(0)
    target_length = target_word_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        if use_char_embedding:
            encoder_output, encoder_hidden = encoder(input_word_tensor[ei], input_char_tensor[ei], encoder_hidden)
        else:
            encoder_output, encoder_hidden = encoder(input_word_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < parameters['teacher_forcing_ratio'] else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if (use_attention):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_word_tensor[di])
            decoder_input = target_word_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if (use_attention):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_word_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

lang_word_indexer = WordIndexer(input_lang, output_lang, device)
lang_char_indexer = CharIndexer(input_lang, output_lang, device)
file_path = output_dir + '/' + parameters['plot_file']

def trainIters(encoder,
               decoder,
               n_iters=parameters['train_iters'],
               print_every=parameters['print_every'],
               plot_every=parameters['plot_every'],
               learning_rate=parameters['learning_rate']):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    random_pairs = [random.choice(pairs) for i in range(n_iters)]
    training_word_pairs = [lang_word_indexer.pairToTensors(pair) for pair in random_pairs]
    if use_char_embedding:
        input_char_indexes = [lang_char_indexer.sentenceToTensor(input_lang, pair[0]) for pair in random_pairs]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_word_pair = training_word_pairs[iter - 1]
        input_word_tensor = training_word_pair[0]
        target_word_tensor = training_word_pair[1]
        if use_char_embedding:
            input_char_tensor = input_char_indexes[iter - 1]

        loss = train(input_word_tensor, target_word_tensor, input_char_tensor,
                     encoder, decoder, encoder_optimizer, decoder_optimizer,
                     criterion, max_length=MAX_LENGTH)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    savePlot(file_path, plot_losses)

def evaluate(encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_words = lang_word_indexer.sentenceToTensor(input_lang, sentence)
        if use_char_embedding:
            input_chars = lang_char_indexer.sentenceToTensor(input_lang, sentence)
        input_length = input_words.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            if use_char_embedding:
                encoder_output, encoder_hidden = encoder(input_words[ei], input_chars[ei], encoder_hidden)
            else:
                encoder_output, encoder_hidden = encoder(input_words[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length=MAX_LENGTH)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

if parameters['train']:
    if use_char_embedding:
        encoder = CharEncoderRNN(input_lang.n_words,
                                 parameters['hidden_size'],
                                 input_lang.n_chars,
                                 parameters['char_embedding_size'],
                                 parameters['char_representation_size'],
                                 device)
    else:
        encoder = EncoderRNN(input_lang.n_words, parameters['hidden_size'], device)
    if use_attention:
        decoder = AttnDecoderRNN(parameters['hidden_size'],
                                 output_lang.n_words,
                                 device,
                                 max_length=MAX_LENGTH,
                                 dropout_p=0.1)
    else:
        decoder = DecoderRNN(parameters['hidden_size'], output_lang.n_words, device)
    trainIters(encoder, decoder)
    torch.save(encoder, output_dir + '/' + parameters['encoder_file'])
    torch.save(decoder, output_dir + '/' + parameters['decoder_file'])
else:
    encoder = torch.load(output_dir + '/' + parameters['encoder_file'])
    decoder = torch.load(output_dir + '/' + parameters['decoder_file'])

    if use_char_embedding and not hasattr(encoder, 'char_embedding'):
        raise Exception('Saved encoder doesn\'t use char embedding. Please run with --use-char-embedding=false')
    if not use_char_embedding and hasattr(encoder, 'char_embedding'):
        raise Exception('Saved encoder uses char embedding. Please run without --use-char-embedding=false')

    if use_attention and not hasattr(decoder, 'attn'):
        raise Exception('Saved decoder doesn\'t use attention. Please run with --use-attention=false')
    if not use_attention and hasattr(decoder, 'attn'):
        raise Exception('Saved decoder uses attention. Please run without --use-attention=false')


evaluateRandomly(encoder, decoder)

def evaluateAndShowAttention(input_sentence, filename):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, max_length=MAX_LENGTH)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    if use_attention:
        file_path = output_dir + '/' + filename
        saveAttention(file_path, input_sentence, output_words, attentions, clear=True)


evaluateAndShowAttention("elle a cinq ans de moins que moi .", '0')

evaluateAndShowAttention("elle est trop petit .", '1')

evaluateAndShowAttention("je ne crains pas de mourir .", '2')

evaluateAndShowAttention("c est un jeune directeur plein de talent .", '3')
