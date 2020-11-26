#app.py
import json
import sqlite as db
import glob
import helper
import os


from flask import (Flask,send_from_directory,render_template,request)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/setup")
def setup():
  helper.save_model()
  body = {
    "message": "Setup Successfull"
  }
  response = {
    "statusCode": 200,
    "body": json.dumps(body)
  }
  return response

@app.route("/search",methods = ['POST', 'GET'])
def search():
  search_st = request.form.get('searched')
  print(search_st)
  result = helper.get_similar_documents(search_st)
  print(result)
  return render_template('result.html', results = result) 

@app.route("/similar",methods = ['GET'])
def similar():
  doc_name = request.args.get('doc')
  result = helper.get_documents_from_same_cluster(doc_name)
  print(result)
  return render_template('result.html', results = result) 
