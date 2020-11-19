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

corpus = []
index_arr = []

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

def add_documents(stemmed, document_name):
    index_arr.append(document_name)
    corpus.append(" ".join(stemmed))

def preprocess_document(doc_name, doc_type):
    file_path = "Resume/" + doc_name
    if (doc_type == "docx"):
        text = read_docx_document(file_path)
    if (doc_type == "pdf"):
        text = read_pdf_document(file_path)
    tokens = tokenize_words(text)
    words = check_alphanumberic_words(tokens)
    words = remove_stop_words(words)
    stemmed = apply_stemmer(words)
    add_documents(stemmed, doc_name)
    return stemmed

def vector_space_model(url, filename):
  vectorizer = TfidfVectorizer()
  if filename.endswith(".docx"):
    text = read_docx_document(url)
  if filename.endswith(".pdf"):
    text = read_pdf_document(url)
  text = read_docx_document(url)
  tokens = tokenize_words(text)
  words = check_alphanumberic_words(tokens)
  words = remove_stop_words(words)
  stemmed = apply_stemmer(words)
  X = vectorizer.fit_transform([" ".join(stemmed)])
  doc_term_matrix = X.todense()
  tf_idf_data = pd.DataFrame(doc_term_matrix, 
                 columns=vectorizer.get_feature_names(), 
                index=[filename])
  return tf_idf_data.to_dict(orient='records')
# # Create the Document vector space Matrix
# vectorizer = TfidfVectorizer()

# #Sample query
# text = read_docx_document("Job Description/CDL - EVP Head of Asset Mgt.docx")
# tokens = tokenize_words(text)
# words = check_alphanumberic_words(tokens)
# words = remove_stop_words(words)
# stemmed = apply_stemmer(words)
# add_documents(stemmed, "Query")

# X = vectorizer.fit_transform(corpus)
# print(X.shape)
# doc_term_matrix = X.todense()
# tf_idf_data = pd.DataFrame(doc_term_matrix, 
#                  columns=vectorizer.get_feature_names(), 
#                 index=index_arr)
