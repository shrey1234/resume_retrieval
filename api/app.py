#app.py
import json
import sqlite as db

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to application"

@app.route("/setup")
def setup():
  db.save_model()
  body = {
    "message": "Setup Successfull"
  }
  response = {
    "statusCode": 200,
    "body": json.dumps(body)
  }
  return response

@app.route("/retrieve")
def retrieve():
  return "Hello World!"