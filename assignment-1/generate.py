###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--prompt', type=str,
                    help="input prompt, if any")
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

if args.prompt:
    prompt = args.prompt.split()
    input = torch.tensor([], dtype=torch.long).to(device)
    for i in range(0, len(prompt)):
        input = torch.cat((input, torch.tensor([corpus.dictionary.word2idx[prompt[i]]]).to(device)))
    input = input[:model.seq_size].view(1, -1)
else:
    input = torch.randint(ntokens, (1, model.seq_size), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    outf.write('==== Prompt ==========================\n' )
    for i in range(0, input.size(1)):
        outf.write(corpus.dictionary.idx2word[input[0][i]] + ' ')
    outf.write('\n\n==== Output ==========================\n\n' )
    for i in range(0, input.size(1)):
        outf.write(corpus.dictionary.idx2word[input[0][i]] + ' ')
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output = model(input)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input = torch.cat((input.view(-1)[1:], torch.tensor([word_idx]).to(device))).view(1, -1)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write('\n' if word == '<eos>' else word + ' ')

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
