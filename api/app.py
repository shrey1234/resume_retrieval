#app.py
import json
import sqlite as db
import helper

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to application"

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

@app.route("/search")
def search():
  result = helper.get_similar_documents("accountant")
  body = {
    "message": "Setup Successfull",
    "result": result
  }
  response = {
    "statusCode": 200,
    "body": json.dumps(body)
  }
  return response