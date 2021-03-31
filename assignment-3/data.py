import unicodedata
import re
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

class Lang:
    def __init__(self, name):
        self.name = name

        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        self.char2index = {}
        self.char2count = {}
        self.index2char = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_chars = 2  # Count SOS and EOS
        self.addChar(' ')

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Initializing Lang...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

class WordIndexer():
    def __init__(self, lang1, lang2, device):
        self.lang1 = lang1
        self.lang2 = lang2
        self.device = device

    def sentenceToIndex(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def sentenceToTensor(self, lang, sentence):
        indexes = self.sentenceToIndex(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def pairToTensors(self, pair):
        input_tensor = self.sentenceToTensor(self.lang1, pair[0])
        target_tensor = self.sentenceToTensor(self.lang2, pair[1])
        return (input_tensor, target_tensor)

class CharIndexer():
    def __init__(self, lang1, lang2, device):
        self.lang1 = lang1
        self.lang2 = lang2
        self.device = device

    def wordToIndex(self, lang, word):
        return [[lang.char2index[char] for char in word]]

    def wordToTensor(self, lang, word):
        return torch.tensor(self.wordToIndex(lang, word), dtype=torch.long, device=self.device)

    def sentenceToTensor(self, lang, sentence):
        indexes = [self.wordToIndex(lang, word) for word in sentence.split(' ')]
        indexes.append([[EOS_token]])
        return indexes

    def sentenceToTensor(self, lang, sentence):
        indexes = [self.wordToTensor(lang, word) for word in sentence.split(' ')]
        indexes.append(torch.tensor([[EOS_token]], dtype=torch.long, device=self.device))
        return indexes
