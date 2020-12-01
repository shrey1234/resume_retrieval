# resume_retrieval
CMPE-256 Project

Installation steps
Db Setup
1. sqlite3 ResumeRetrieval.db
2. CREATE TABLE IF NOT EXISTS resumes (id integer PRIMARY KEY, cluster integer, resume_name text NOT NULL, url text NOT NULL, vector_model json );
3. commit;

Run Application
1. pip install numpy
2. pip install pandas
3. pip install docx
4. pip install PyPDF2
5. pip install nltk
6. pip install requests
7. pip install docx2txt
8. pip install matplotlib
9. pip install sklearn
10. pip install seaborn
11. pip install glob
12. pip install sqlite
13. export FLASK_APP=app.py
14. flask run
Application will run at localhost:5000

Setup Call
1. localhost:5000/setup

Go to home page for search - localhost:5000

