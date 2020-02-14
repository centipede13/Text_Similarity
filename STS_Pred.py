import collections

import numpy as np
import pandas as pd

import nltk, string
from nltk import word_tokenize # Convert paragraph in tokens
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

text_data = pd.read_csv("Text_Similarity_Dataset.csv")

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''removing punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1] #p [0,1] is the positions in the matrix for the similarity

similarity = [] 
for i in text_data.index:
    
        sent1 = text_data['text1'][i]
        sent2 = text_data['text2'][i]

        similarity.append(cosine_sim(sent1,sent2))

final_score = pd.DataFrame({'Unique_ID':text_data.Unique_ID,
                     'Similarity_score':similarity})

final_score.to_csv('final_score.csv',index=False)