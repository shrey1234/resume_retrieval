import sqlite3
import os
import helper 
import json

# CREATE TABLE resumes (id integer PRIMARY KEY, resume_name text NOT NULL, url text NOT NULL, vector_model json, cluster text );

def save_model():
  conn = sqlite3.connect('ResumeRetrieval.db')
  dir = "resumes"
  values = []
  i = 1
  for filename in os.listdir(dir):
    print("resumes/"+filename)
    vector_model = helper.vector_space_model("resumes/"+filename, filename)
    # cluster = method for cluster assignment
    print(vector_model)
    values.append((i, filename, "resumes/"+filename, json.dumps(vector_model[0])))
    i += 1
  result = conn.executemany('INSERT INTO resumes (id, resume_name, url, vector_model) VALUES (?,?,?,?)', values)
  conn.commit()
  conn.close()
  return "Successfully inserted"