import os

if not os.path.isfile('./data/eng-fra.txt'):
  os.system('wget -P ./ https://download.pytorch.org/tutorial/data.zip')
  os.system('unzip ./data.zip -d ./')
  os.system('rm ./data.zip')
