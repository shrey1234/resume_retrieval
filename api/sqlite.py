import sqlite3
import json
import numpy as np

def create_table(conn):
  conn.execute('DROP TABLE resumes;')
  conn.execute('CREATE TABLE IF NOT EXISTS resumes (id integer PRIMARY KEY, cluster integer, resume_name text NOT NULL, url text NOT NULL, vector_model json );')
  return True

def save_in_db(result, corpus_name,cluster):
  sqlite3.register_adapter(np.int32, int)
  conn = sqlite3.connect('ResumeRetrieval.db')
  create_table(conn)
  values = []
  i = 1
  for value in result:
    values.append((i,cluster[i-1],corpus_name[i-1], corpus_name[i-1], json.dumps(value)))
    i += 1
  print("Executing query")
  result = conn.executemany('INSERT INTO resumes (id,cluster,resume_name, url,vector_model) VALUES (?,?,?,?,?)', values)
  conn.commit()
  conn.close()
  return "Successfully inserted"


def get_all_records():
  conn = sqlite3.connect('ResumeRetrieval.db')
  cursor = conn.cursor()
  cursor.execute('SELECT id, resume_name, url, vector_model, cluster FROM resumes;')
  result = cursor.fetchall()
  return result

def get_cluster_number(doc_name):
  conn = sqlite3.connect('ResumeRetrieval.db')
  cursor = conn.cursor()
  cursor.execute('SELECT * FROM resumes WHERE resume_name=?;',(doc_name,))
  result = cursor.fetchone()
  return result[1]

def get_docs_with_same_cluster(cluster_id):
  conn = sqlite3.connect('ResumeRetrieval.db')
  cursor = conn.cursor()
  rows = cursor.execute('SELECT * FROM resumes WHERE cluster=?;',(cluster_id,))
  matched_documents = []

  for row in rows:
     matched_documents.append({'name': row[2], 'url': row[3]})
  
  return matched_documents




