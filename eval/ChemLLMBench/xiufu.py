import nltk, os
path = os.path.expanduser('~/nltk_data')
nltk.data.path.append(path)
nltk.download('wordnet', download_dir=path)
nltk.download('omw-1.4', download_dir=path)

from nltk.corpus import wordnet as wn
print(wn.synsets("dog"))
