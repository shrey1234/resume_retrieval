# resume_retrieval
CMPE-256 Project

Setup Commands
Install Db and create Table
1. Install sqlite3 on pc
2. cd api
3. sqlite3 ResumeRetrieval.db
4. CREATE TABLE resumes (id integer PRIMARY KEY, resume_name text NOT NULL, url text NOT NULL, vector_model json, cluster text );

Run Server
1. cd api
2. Install all the packages
3. sls wsgi serve
