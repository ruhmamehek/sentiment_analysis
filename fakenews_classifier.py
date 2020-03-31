import pandas as pd
import nltk
import random
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

df = pd.read_csv("politifact_fake.csv")
# reading data from csv file. df is a dataframe consisting 
# of fake news that we will use to train our model.
dr = pd.read_csv("politifact_real.csv")
# reading data from csv file. dr is a dataframe consisting 
# of real news that we will use to train our model.



tokenizer=RegexpTokenizer(r'\w+')
dr['title']=dr['title'].apply(lambda x: tokenizer.tokenize(x.lower()))
df['title']= df['title'].apply(lambda x: tokenizer.tokenize(x.lower()))
# tokenizing the data, i.e. splitting the title into substrings using a regular expression.


def remove_stopWords(text):
  """
  Removes stop words from the list of words provide
  
  Args:
    text: list of words (strings)
    
    Returns: list of words (strings), after removing stopwords
    
  """
  words = [w for w in text if w not in stopwords.words('english')]
  return words

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    """
     groups different inflected forms of a word together
    
    Args:
      text: list of words (strings)
      
    Returns: list of words (strings), after performing lemmatization
    
    """
  lem_text = [lemmatizer.lemmatize(i) for i in text]
  return (lem_text)

stemmer = PorterStemmer()

def word_stemmer(text):
  
    """
     converts words to its root/base form
    
    Args:
      text: list of words (strings)
      
    Returns: list of words, after stemming each string
    
    """
  
  stem_text = "".join([stemmer.stem(i) for i in text ])
  return stem_text

def remove_punctuation(text):
  
     """
     removes punctuation marks from text
    
    Args:
      text: list of words (strings)
      
    Returns: list of words (strings), after removing strings that were punctuation marks
    """
    
  
  no_punct = "".join([c for c in text if c not in string.punctuation])
  return no_punct


dr['title']= dr['title'].apply(lambda x: remove_stopWords(x))
dr['title']=dr['title'].apply(lambda x: word_lemmatizer(x))


df['title']=df['title'].apply(lambda x: remove_stopWords(x))
df['title']=df['title'].apply(lambda x: word_lemmatizer(x))


arr_real=[]
for i in dr['title']:
  arr_real.append((list(i), 'real'))
  
# adding tuples of the processed real news titles, with the string 'real' to a list


arr_fake=[]
for i in df['title']:
  arr_fake.append((list(i), 'fake'))
# print('arr_fake', arr_fake)

# adding tuples of the processed fake news titles, with the fake 'real' to a list


all_words=[]

for i in dr['title']:
  all_words.append(i)
for i in df['title']:
  all_words.append(i)
all_words = [item for sublist in all_words for item in sublist]

all_titles=[]
all_titles=arr_fake+arr_real
random.shuffle(all_titles)
# making a list of all titles in dataset, and shuffling them


def document_features(document):
    """
    creates a feature set, checking the words in the provided documents 
    and marking their presence as True or False. 
    This function is used in this code to create feature sets.
    
    """"
    document_words = set(document)
    features = {}
    for word in all_words:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in all_titles]
train_set, test_set = featuresets[:800], featuresets[800:]
# dividing dataset into training and dataset

classifier = nltk.NaiveBayesClassifier.train(train_set)
# making a Naive Bayes classifier using our training data

print(nltk.classify.accuracy(classifier, test_set))

headlineTest = input("Enter Headline to classify: ")
print(classifier.classify(document_features(headlineTest)))




