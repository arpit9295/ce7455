import torch
from torch.autograd import Variable
import argparse
from data import get_chunks
from utils import lower_case
import _pickle as cPickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', default='logs')
parser.add_argument('--sentence')
parser.add_argument('--lower', default=True,
                    help="Boolean variable to control lowercasing of words")
parser.add_argument('--model', default='1_CNN.model', help="Model path")
parser.add_argument('--use-gpu', default=True)

mapping_file = './data/mapping.pkl'

parameters = vars(parser.parse_args())

if (parameters['sentence']):
    model_testing_sentences = [parameters['sentence']]
else:
    model_testing_sentences = ['Krishnalal Murthy is a pussy']

lower = parameters['lower']
use_gpu = parameters['use_gpu']

with open(mapping_file, 'rb') as f:
    mappings = cPickle.load(f)
    word_to_id = mappings['word_to_id']
    tag_to_id = mappings['tag_to_id']
    char_to_id = mappings['char_to_id']
    word_embeds = mappings['word_embeds']

# preprocessing
final_test_data = []
for sentence in model_testing_sentences:
    s = sentence.split()
    str_words = [w for w in s]
    words = [word_to_id[lower_case(w, lower) if lower_case(
        w, lower) in word_to_id else '<UNK>'] for w in str_words]

    # Skip characters that are not in the training set
    chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]

    final_test_data.append({
        'str_words': str_words,
        'words': words,
        'chars': chars,
    })

if use_gpu:
    model = torch.load(parameters['output_dir'] +
                       '/' + parameters['model']).cuda()
else:
    model = torch.load(parameters['output_dir'] + '/' + parameters['model'])


predictions = []
unk_id = word_to_id['<UNK>']

print("Prediction:")
print("word : tag")
for data in final_test_data:
    words = data['str_words']
    word_ids = data['words']
    chars2 = data['chars']

    d = {}

    # Padding the each word to max word size of that sentence
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
    for i, c in enumerate(chars2):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = Variable(torch.LongTensor(chars2_mask))

    dwords = Variable(torch.LongTensor(data['words']))

    # We are getting the predicted output from our model
    if use_gpu:
        val, predicted_id = model(
            dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
    else:
        val, predicted_id = model(dwords, chars2_mask, chars2_length, d)

    pred_chunks = get_chunks(predicted_id, tag_to_id)
    temp_list_tags = ['NA']*len(words)
    for p in pred_chunks:
        temp_list_tags[p[1]] = p[0]

    for word, tag, word_id in zip(words, temp_list_tags, word_ids):
        if (word_id == unk_id):
            print('<UNK>')
        else:
            print(word, ':', tag)
    print('\n')
