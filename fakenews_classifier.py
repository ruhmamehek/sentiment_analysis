import pandas as pd
import nltk
import random
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')



df = pd.read_csv("politifact_fake.csv")
dr = pd.read_csv("politifact_real.csv")


tokenizer=RegexpTokenizer(r'\w+')
dr['title']=dr['title'].apply(lambda x: tokenizer.tokenize(x.lower()))
df['title']= df['title'].apply(lambda x: tokenizer.tokenize(x.lower()))


def remove_stopWords(text):
  words = [w for w in text if w not in stopwords.words('english')]
  return words

lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
  lem_text = [lemmatizer.lemmatize(i) for i in text]
  return (lem_text)

stemmer = PorterStemmer()
def word_stemmer(text):
  stem_text = "".join([stemmer.stem(i) for i in text ])
  return stem_text
def remove_punctuation(text):
  no_punct = "".join([c for c in text if c not in string.punctuation])
  return no_punct

# print("preprocessed real titles ")
# print(dr['title'])
# print("preprocessed fake titles ")
# print(df['title'])

# dr['title']=dr['title'].apply(lambda x: remove_punctuation(x))
dr['title']= dr['title'].apply(lambda x: remove_stopWords(x))
dr['title']=dr['title'].apply(lambda x: word_lemmatizer(x))
# dr['title']=dr['title'].apply(lambda x: word_stemmer(x))

# df['title']=dr['title'].apply(lambda x: remove_punctuation(x))
df['title']=df['title'].apply(lambda x: remove_stopWords(x))
df['title']=df['title'].apply(lambda x: word_lemmatizer(x))
# df['title']=df['title'].apply(lambda x: word_stemmer(x))

arr_real=[]
for i in dr['title']:
  arr_real.append((list(i), 'real'))

# print('arr_real', arr_real)
arr_fake=[]
for i in df['title']:
  arr_fake.append((list(i), 'fake'))
# print('arr_fake', arr_fake)


all_words=[]

for i in dr['title']:
  all_words.append(i)
for i in df['title']:
  all_words.append(i)
all_words = [item for sublist in all_words for item in sublist]

all_titles=[]
all_titles=arr_fake+arr_real
random.shuffle(all_titles)

# print("all titles", all_titles)
# print(all_words)


def document_features(document):
    document_words = set(document)
    features = {}
    for word in all_words:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in all_titles]
train_set, test_set = featuresets[:800], featuresets[800:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

headlineTest = input("Enter Headline to classify: ")
print(classifier.classify(document_features(headlineTest)))




