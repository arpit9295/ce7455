from __future__ import print_function
import pandas as pd
import argparse
import numpy as np
import codecs
import os

import torch
from torch.autograd import Variable
import time
import _pickle as cPickle

import matplotlib.pyplot as plt
import sys

from models import BiLSTM_CRF
from data import load_sentences, update_tag_scheme, prepare_dataset, \
                 to_word_mapping, to_char_mapping, to_tag_mapping, \
                 get_chunks
from utils import adjust_learning_rate, get_num_params, Unbuffered
from early_stopping import EarlyStopping

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-pastel')


# ##### Define constants and paramaters

parser = argparse.ArgumentParser()
# parameters for the Model
parser.add_argument('--train', default="./data/eng.train", help="Path to train file")
parser.add_argument('--dev', default="./data/eng.testa", help="Path to test file")
parser.add_argument('--test', default="./data/eng.testb", help="Path to dev file")
parser.add_argument('--tag_scheme', default="BIOES", help="BIO or BIOES")
parser.add_argument('--lower', default=True, help="Boolean variable to control lowercasing of words")
parser.add_argument('--zeros', default=True, help="Boolean variable to control replacement of  all digits by 0")
parser.add_argument('--char-dim', type=int, default=30, help="Char embedding dimension")
parser.add_argument('--word-dim', type=int, default=100, help="Token embedding dimension")
parser.add_argument('--word-lstm-dim', type=int, default=200, help="Token LSTM hidden layer size")
parser.add_argument('--word-bidirect', default=True, help="Use a bidirectional LSTM for words")
parser.add_argument('--embedding-path', default="./data/glove.6B.100d.txt", help="Location of pretrained embeddings")
parser.add_argument('--all-emb', type=int, default=1, help="Load all embeddings")
parser.add_argument('--crf', type=int, default=1, help="Use CRF (0 to disable)")
parser.add_argument('--dropout', type=int, default=0.5, help="Droupout on the input (0 = no dropout)")
parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to run")
parser.add_argument('--weights', default="", help="path to Pretrained for from a previous run")
parser.add_argument('--name', default="self-trained-model", help="Model name")
parser.add_argument('--gradient-clip', default=5.0)
parser.add_argument('--char-mode', default="CNN")
parser.add_argument('--encoder-mode', default="LSTM")
parser.add_argument('--use-gpu', default=True)
parser.add_argument('--output-dir', default='logs')
parser.add_argument('--plot-every', type=int, default=2000)
parser.add_argument('--eval-every', type=int, default=4, help="Number of epochs to run")

models_path = "./models/"  # path to saved models
parameters = vars(parser.parse_args())

if not os.path.exists(parameters['output_dir']):
    os.system('mkdir ' + parameters['output_dir'])

f = open(parameters['output_dir'] + '/out.txt', 'w')
sys.stdout = Unbuffered(f) # Change the standard output to the file we created.

if torch.cuda.is_available():
    if not parameters['use_gpu']:
        print("WARNING: You have a CUDA device, so you should probably run with --use_gpu")


parameters['reload'] = "./models/self-trained-model"

# Constants


# paths to files
# To stored mapping file
mapping_file = './data/mapping.pkl'

# To stored model
name = parameters['name']
model_name = models_path + name  # get_name(parameters)

if not os.path.exists(models_path):
    os.makedirs(models_path)

# ##### Load data and preprocess

train_sentences = load_sentences(parameters['train'], parameters['zeros'])
test_sentences = load_sentences(parameters['test'], parameters['zeros'])
dev_sentences = load_sentences(parameters['dev'], parameters['zeros'])

update_tag_scheme(train_sentences, parameters['tag_scheme'])
update_tag_scheme(dev_sentences, parameters['tag_scheme'])
update_tag_scheme(test_sentences, parameters['tag_scheme'])

print(train_sentences[0])
print(dev_sentences[0])
print(test_sentences[0])


dico_words, word_to_id, id_to_word = to_word_mapping(
    train_sentences, parameters['lower'])
dico_chars, char_to_id, id_to_char = to_char_mapping(train_sentences)
dico_tags,  tag_to_id,  id_to_tag = to_tag_mapping(train_sentences)


# ##### Preparing final dataset

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
print("{} / {} / {} sentences in train / dev / test.".format(len(train_data),
                                                            len(dev_data), len(test_data)))


# ##### Load Word Embeddings


all_word_embeds = {}
for i, line in enumerate(codecs.open(parameters['embedding_path'], 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

# Intializing Word Embedding Matrix
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06),
                                (len(word_to_id), parameters['word_dim']))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


# ##### Storing Processed Data for Reuse


with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

print('word_to_id: ', len(word_to_id))

# #### Evaluation schemes: Forward pass and Viterbi algorithm

# ### Details of the Model

# ##### Main Model Implementation

# The get_lstm_features function returns the LSTM's tag vectors. The function performs all the steps mentioned above for the model.
#
# Steps:
# 1. It takes in characters, converts them to embeddings using our character CNN.
# 2. We concat Character Embeeding with glove vectors, use this as features that we feed to Bidirectional-LSTM.
# 3. The Bidirectional-LSTM generates outputs based on these set of features.
# 4. The output are passed through a linear layer to convert to tag space.

# ### Evaluation

# ##### Helper functions for evaluation


def evaluating(model,
               datas,
               best_F,
               dataset="Train",
               char_mode=parameters['char_mode'],
               use_gpu=parameters['use_gpu']):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates
    1) Flag to save the model
    2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    # Initializations
    prediction = []  # A list that stores predicted tags
    save = False  # Flag that tells us if the model needs to be saved
    new_F = 0.0  # Variable to store the current F1-Score (may not be the best)
    correct_preds, total_correct, total_preds = 0., 0., 0.  # Count variables

    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']

        if char_mode == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros(
                (len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if char_mode == 'CNN':
            d = {}

            # Padding the each word to max word size of that sentence
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros(
                (len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))

        # We are getting the predicted output from our model
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out

        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks = set(get_chunks(ground_truth_id, tag_to_id))
        lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id))

        # Updating the count variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    # Calculating the F1-Score
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    new_F = 2 * p * r / (p + r) if correct_preds > 0 else 0
    new_acc = p

    print("{}: new_F: {} best_F: {} new_acc:{} ".format(dataset, new_F, best_F, p))

    # If our current F1-Score is better than the previous best, we update the best
    # to current F1 and we set the flag to indicate that we need to checkpoint this model

    if new_F > best_F:
        best_F = new_F
        # save=True

    return best_F, new_F, new_acc, save


# ##### Helper function for performing early stopping

# <a name='experiments'></a>
# # 2. EXPERIMENTS
# [back to top](#outline)

# ### (iv) Replace the LSTM-based word-level encoder with a CNN layer (convolutional layer followed by an optional max pooling layer). The CNN layer should have the same output dimensions (out_channels) as the LSTM.

# #### Get CNN Features Function
#
# The get_cnn_features function returns the CNN's tag vectors. The function performs all the steps mentioned above for the model.
#
# Steps:
# 1. It takes in characters, converts them to embeddings using our character CNN.
# 2. We concat Character Embedding with glove vectors, use this as features that we feed to CNN.
# 3. The CNN generates outputs based on these set of features.
# 4. The output are passed through a linear layer to convert to tag space.


# #### Model Class


# #### Create and Train Model Function


def init_model_and_train(label='',
                         crf=parameters['crf'],
                         char_mode=parameters['char_mode'],
                         encoder_mode=parameters['encoder_mode'],
                         use_gpu=parameters['use_gpu'],
                         eval_every = parameters['eval_every'], # Calculate F-1 Score after this many iterations
                         plot_every = parameters['plot_every'],  # Store loss after this many iterations
                         gradient_clip = parameters['gradient_clip'],
                         total_epochs=parameters['epochs'] + 1,
                         output_dir=parameters['output_dir'],
                         embedding_dim=parameters['word_dim'],
                         hidden_dim=parameters['word_lstm_dim']):
    # Create model
    model = BiLSTM_CRF(vocab_size=len(word_to_id),
                       tag_to_ix=tag_to_id,
                       embedding_dim=embedding_dim,
                       hidden_dim=hidden_dim,
                       use_gpu=use_gpu,
                       char_to_ix=char_to_id,
                       pre_word_embeds=word_embeds,
                       use_crf=crf,
                       char_mode=char_mode,
                       encoder_mode=encoder_mode)

    # Enable GPU
    if use_gpu:
        model.cuda()

    print(f"Char mode: {char_mode}, Encoder mode: {encoder_mode}")

    # Training parameters
    learning_rate = 0.015
    momentum = 0.9
    decay_rate = 0.05
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)

    # Variables which will used in training process
    losses = []  # list to store all losses
    loss = 0.0  # Loss Initializatoin
    best_dev_F = -1.0  # Current best F-1 Score on Dev Set
    best_test_F = -1.0  # Current best F-1 Score on Test Set
    best_train_F = -1.0  # Current best F-1 Score on Train Set
    all_F = [[0, 0, 0]]  # List storing all the F-1 Scores
    all_acc = [[0, 0, 0]]  # List storing all the Accuracy Scores
    count = 0  # Counts the number of iterations
    train_length = len(train_data)

    # Define early stopping
    es = EarlyStopping(patience=3, mode='max')

    # eval_every = 1

    tr = time.time()
    model.train(True)
    for epoch in range(1, total_epochs):
        print(f'Epoch {epoch}:')
        for i, index in enumerate(np.random.permutation(train_length)):
            # for i, index in enumerate(np.random.permutation(eval_every)):
            count += 1
            data = train_data[index]

            # gradient updates for each data entry
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']

            if char_mode == 'LSTM':
                chars2_sorted = sorted(
                    chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for i, ci in enumerate(chars2):
                    for j, cj in enumerate(chars2_sorted):
                        if ci == cj and not j in d and not i in d.values():
                            d[j] = i
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros(
                    (len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            if char_mode == 'CNN':

                d = {}

                # Padding the each word to max word size of that sentence
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros(
                    (len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)

            # we calculate the negative log-likelihood for the predicted tags using the predefined function
            if use_gpu:
                neg_log_likelihood = model.get_neg_log_likelihood(
                    sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length, d)
            else:
                neg_log_likelihood = model.get_neg_log_likelihood(
                    sentence_in, targets, chars2_mask, chars2_length, d)

            loss += neg_log_likelihood.item() / len(data['words'])
            neg_log_likelihood.backward()

            # we use gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # Storing loss
            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                if losses == []:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

        # Evaluating on Train, Test, Dev Sets
        if (epoch > 20) or (epoch % eval_every == 0):
            print(f'Evaluating on Train, Test, Dev Sets at count={count}')
            model.train(False)
            best_train_F, new_train_F, new_train_acc, _ = evaluating(model, train_data, best_train_F, "Train", char_mode=char_mode, use_gpu=use_gpu)
            best_dev_F, new_dev_F, new_dev_acc, save = evaluating(model, dev_data, best_dev_F, "Dev", char_mode=char_mode, use_gpu=use_gpu)
            if save:
                print("Saving Model to ", model_name)
                torch.save(model.state_dict(), model_name)
            best_test_F, new_test_F, new_test_acc, _ = evaluating(model, test_data, best_test_F, "Test", char_mode=char_mode, use_gpu=use_gpu)

            all_F.append([new_train_F, new_dev_F, new_test_F])
            all_acc.append([new_train_acc, new_dev_acc, new_test_acc])

            model.train(True)

        if (epoch > 20 or epoch % eval_every == 0) and es.step(all_F[-1][1]):
            print(f'Early stopping: epoch={epoch}, count={count}, new_acc_F={all_acc[-1][1]}')
            break  # early stopping criterion is met, we can stop now

        # Performing decay on the learning rate
        adjust_learning_rate(optimizer, lr=learning_rate/(1+decay_rate*count/len(train_data)))

    print(f'{(time.time() - tr) / 60} minutes')

    torch.save(model, output_dir + '/' + label + '.model')

    plt.figure(0)
    plt.plot(losses)
    plt.savefig(output_dir + '/' + label + '_appended.png', transparent=True)

    plt.figure(1)
    plt.clf()
    plt.plot(losses)
    plt.savefig(output_dir + '/' + label + '.png', transparent=True)

    return all_F


# 1-layer CNN word + CNN char
print('\n1-layer CNN word + CNN char')
all_F_1_CNN = init_model_and_train(label='1_CNN',
                                   char_mode = "CNN",
                                   encoder_mode = "CNN",
                                   crf = 1)
print('All F1 =', all_F_1_CNN)


# 1-layer CNN word + LSTM char
print('\n1-layer CNN word + LSTM char')
all_F_1_LSTM = init_model_and_train(label='1_LSTM',
                                    char_mode = "LSTM",
                                    encoder_mode = "CNN",
                                    crf = 1)
print('All F1 =', all_F_1_LSTM)


# 2-layer CNN word + CNN char
print('\n2-layer CNN word + CNN char')
all_F_2_CNN = init_model_and_train(label='2_CNN',
                                   char_mode = "CNN",
                                   encoder_mode = "CNN2",
                                   crf = 1)
print('All F1 =', all_F_2_CNN)


# 3-layer CNN word + CNN char
print('\n3-layer CNN word + CNN char')
all_F_3_CNN = init_model_and_train(label='3_CNN',
                                   char_mode = "CNN",
                                   encoder_mode = "CNN3",
                                   crf = 1)
print('All F1 =', all_F_3_CNN)


# Dialeted CNN word + CNN Char
print('\nDialeted CNN word + CNN Char')
all_F_DILATED = init_model_and_train(label='DILATED',
                                     char_mode = "CNN",
                                     encoder_mode = "CNN_DILATED",
                                     crf = 1)
print('All F1 =', all_F_DILATED)


# Dialeted CNN word + CNN Char + Softmax instead of CRF
print('\nDialeted CNN word + CNN Char + Softmax instead of CRF')
all_F_DILATED_SOFTMAX = init_model_and_train(label='DI,LATED_SOFTMAX',
                                             char_mode = "CNN",
                                             encoder_mode = "CNN_DILATED",
                                             crf = 0)
print('All F1 =', all_F_DILATED_SOFTMAX)


# CNN-CNN
cnn_cnn = BiLSTM_CRF(vocab_size=len(word_to_id),
                     tag_to_ix=tag_to_id,
                     char_to_ix=char_to_id,
                     pre_word_embeds=word_embeds,
                     char_mode='CNN',
                     encoder_mode='CNN')

# LSTM-CNN
lstm_cnn = BiLSTM_CRF(vocab_size=len(word_to_id),
                      tag_to_ix=tag_to_id,
                      char_to_ix=char_to_id,
                      pre_word_embeds=word_embeds,
                      char_mode='LSTM',
                      encoder_mode='CNN')


# #### 1. `CNN char-level encoder` vs `LSTM char-level encoder` (both using `Single CNN word-level encoder`)
#
# Results and number of parameters in each model.


# Get all val scores of each model
val_cnn_1 = [f1[1] for f1 in all_F_1_CNN]
val_lstm_1 = [f1[1] for f1 in all_F_1_LSTM]

# Get best val score index of each model
best_val_cnn_1_index = val_cnn_1.index(max(val_cnn_1))
best_val_lstm_1_index = val_lstm_1.index(max(val_lstm_1))

# Get test score w.r.t best validation score of each model
test_cnn_1 = all_F_1_CNN[best_val_cnn_1_index][2]
test_lstm_1 = all_F_1_LSTM[best_val_lstm_1_index][2]

# Get best val score of each mode
val_cnn_1 = max(val_cnn_1)
val_lstm_1 = max(val_lstm_1)

# Construct results table
df = pd.DataFrame()
df['model'] = ['cnn_cnn', 'lstm_cnn']
df['parameters'] = [get_num_params(cnn_cnn), get_num_params(lstm_cnn)]
df['val'] = [val_cnn_1, val_lstm_1]
df['test'] = [test_cnn_1, test_lstm_1]

print('CNN char-level encoder vs LSTM char-level encoder (both using Single CNN word-level encoder)\n')
print(df)


# #### 2. `Single-layer CNN word-level encoder` vs `Multi-layer CNN word-level encoder` (all using `CNN char-level encoder`)


# Get all val scores of each model
val_cnn_1 = [f1[1] for f1 in all_F_1_CNN]
val_cnn_2 = [f1[1] for f1 in all_F_2_CNN]
val_cnn_3 = [f1[1] for f1 in all_F_3_CNN]

# Get best val score index of each model
best_val_cnn_1_index = val_cnn_1.index(max(val_cnn_1))
best_val_cnn_2_index = val_cnn_2.index(max(val_cnn_2))
best_val_cnn_3_index = val_cnn_3.index(max(val_cnn_3))

# Get test score w.r.t best validation score of each model
test_cnn_1 = all_F_1_CNN[best_val_cnn_1_index][2]
test_cnn_2 = all_F_2_CNN[best_val_cnn_2_index][2]
test_cnn_3 = all_F_3_CNN[best_val_cnn_3_index][2]

# Get best val score of each mode
val_cnn_1 = max(val_cnn_1)
val_cnn_2 = max(val_cnn_2)
val_cnn_3 = max(val_cnn_3)

# Construct results table
df = pd.DataFrame()
df['model'] = ['cnn_1', 'cnn_2', 'cnn_3']
df['val'] = [val_cnn_1, val_cnn_2, val_cnn_3]
df['test'] = [test_cnn_1, test_cnn_2, test_cnn_3]

print('Single-layer CNN word-level encoder vs Multi-layer CNN word-level encoder (all using CNN char-level encoder)\n')
print(df)
