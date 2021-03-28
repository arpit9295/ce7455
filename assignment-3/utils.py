import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy.core.fromnumeric import transpose

plt.switch_backend('agg')
plt.rcParams['figure.dpi'] = 200

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def savePlot(filePath, points, figure=1, clear=False):
    plt.figure(figure)
    if clear:
        plt.clf()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(filePath, transparent=True)

def saveAttention(filePath, input_sentence, output_words, attentions, figure=2, clear=False):
    # Set up figure with colorbar
    fig = plt.figure(figure)
    if clear:
        plt.clf()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='summer')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig(filePath, transparent=True)
