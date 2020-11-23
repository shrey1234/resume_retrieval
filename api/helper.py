import nltk
import docx
import numpy as np
import pandas as pd
import PyPDF2
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import docx2txt
from io import BytesIO
import os
import sqlite as db
import pickle
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 
from sklearn.datasets.samples_generator import make_blobs


#Helper function to read document from disk and return the text
def read_docx_document(filepath):
  doc = docx.Document(filepath)
  #Merged paragraphs into a single text
  text = ""
  for i in doc.paragraphs:
    text += " "+ i.text
  return text


def read_pdf_document(file_path):
    pdfFile = open(file_path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFile)
    text = ""
    for i in range(pdfReader.numPages):
        page = pdfReader.getPage(i)
        text = text + page.extractText()
    return text

#Tokenizez the words in the text
def tokenize_words(text):
    return nltk.word_tokenize(text)

#Remove stopwords
def remove_stop_words(word_tokens):
    stop_words = set(stopwords.words('english')) 
    filtered_words = [w for w in word_tokens if not w in stop_words] 
    filtered_words = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_words.append(w) 
    return filtered_words


#Checking for alpha numeric words
def check_alphanumberic_words(tokens):
    return [word for word in tokens if word.isalpha()]

#Stemming words
def apply_stemmer(words):
    porter = PorterStemmer()
    return [porter.stem(word) for word in words]

def plot_frequency_distribution(stemmed_words):
    english_stop_words= stopwords.words('english')
    clean_tokens = stemmed_words[:]
    for token in stemmed_words:
        if token in english_stop_words:
            clean_tokens.remove(token)

    freq = nltk.FreqDist(clean_tokens)
    freq.plot(20, cumulative=False)

def preprocess_document(doc_name):
    print(doc_name)  
    file_path = "resumes/" + doc_name
    text = ""
    if doc_name.endswith(".docx"):
      text = read_docx_document(file_path)
    # if doc_name.endswith(".pdf"):
    #   text = read_pdf_document(file_path)
    tokens = tokenize_words(text)
    words = check_alphanumberic_words(tokens)
    words = remove_stop_words(words)
    stemmed = {}
    if(len(words) > 0):
      stemmed = apply_stemmer(words)
    return stemmed

def create_tf_idf_vector():
  return TfidfVectorizer()

def save_vectorizer_model(vectorizer):
  with open('resume_vectorizer.pk', 'wb') as fin:
      pickle.dump(vectorizer, fin)
  return True

def save_model():
  corpus = []
  corpus_name = []
  cluster= []

  vectorizer = create_tf_idf_vector()
  for filename in os.listdir("resumes"):
    if filename.endswith(".docx"):
      stemmed = preprocess_document(filename)
      corpus_name.append(filename)
      corpus.append(" ".join(stemmed))
  X = vectorizer.fit_transform(corpus)
  doc_term_matrix = X.todense()
  tf_idf_data = pd.DataFrame(doc_term_matrix, 
                  columns=vectorizer.get_feature_names(), 
                  index=corpus_name)

  #saving cluster into db
  kmean=KMeans(n_clusters=3)
  kmean.fit(tf_idf_data)
  cluster=kmean.labels_
  result = tf_idf_data.to_dict(orient='records')
  db.save_in_db(result, corpus_name,cluster)
  return save_vectorizer_model(vectorizer)

def get_similar_documents(query):
  vectorizer_new = pickle.load(open("resume_vectorizer.pk", "rb"))
  tokens = tokenize_words(query)
  words = check_alphanumberic_words(tokens)
  words = remove_stop_words(words)
  stemmed = apply_stemmer(words)
  query_features = vectorizer_new.transform([" ".join(stemmed)])
  doc_term_matrix_query = query_features.todense()
  tf_idf_data_query = pd.DataFrame(doc_term_matrix_query, 
                 columns=vectorizer_new.get_feature_names(), 
                index=["Query"])
  records = db.get_all_records()
  vector_arr = []
  for record in records:
    vector_arr.append(json.loads(record[3]))
  b = pd.DataFrame.from_records(vector_arr)
  similarity_test = cosine_similarity(tf_idf_data_query[0:1], b)
  matched_documents = []
  index = 0
  for document in similarity_test[0]:
      if document > 0.1:
          matched_documents.append({'name': records[index][1], 'url': records[index][2]})
          index += 1
  return matched_documents
