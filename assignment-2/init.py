import os

if not os.path.exists('./data'):
  os.system('mkdir data')
if not os.path.isfile('./data/eng.testa'):
  os.system('wget -P ./data https://raw.githubusercontent.com/TheAnig/NER-LSTM-CNN-Pytorch/master/data/eng.testa')
  os.system('wget -P ./data https://raw.githubusercontent.com/TheAnig/NER-LSTM-CNN-Pytorch/master/data/eng.testb')
  os.system('wget -P ./data https://raw.githubusercontent.com/TheAnig/NER-LSTM-CNN-Pytorch/master/data/eng.train')
  os.system('wget -P ./data https://raw.githubusercontent.com/TheAnig/NER-LSTM-CNN-Pytorch/master/data/eng.train54019')
  os.system('wget -P ./data https://raw.githubusercontent.com/TheAnig/NER-LSTM-CNN-Pytorch/master/data/mapping.pkl')
if not os.path.isfile('./data/glove.6B.100d.txt'):
  os.system('wget -P ./data http://nlp.stanford.edu/data/glove.6B.zip')
  os.system('unzip ./data/glove.6B.zip -d ./data')
  os.system('rm ./data/glove.6B.zip')
  os.system('rm ./data/glove.6B.300d.txt')
  os.system('rm ./data/glove.6B.50d.txt')
  os.system('rm ./data/glove.6B.200d.txt')
