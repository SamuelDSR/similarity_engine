## Similarity engine

A simple similarity engine that can match the pseudonym and the real name of an artist.

It first get the answer returned from [duckduckgo](https://duckduckgo.com/api), then use the abstract and abstract_url
of the answers to build a corpus.

The algorithm is based on the **tfidf** of the *real names* and *pseudonym*, you may
aslo need to adjsut the weight of words from abstract_url in the corpus to get better match 
results

### Requirements

1. Python3.x
2. The packages listed in requirements.txt, you can install all packages by: `pip install -r requirements.txt`

### Usuage

1. Put the pseudonym and real names you want to match as two separted files:
ex, pesudonym.csv and real_name.csv `python similarity.py pseudonym.csv real_name.csv`

### API
After calculating the features and similarity scores, you can play with it by using the simple api attached.
Just run `python dash_web.app`

There are two different ways to access the api.

1. GET `http://<your server public ip>:<port you chose>/similarity`
the parameters are pseudonym=? and real_name=?


2. Go to web app `http://<your server public ip>:<port you chose>`

**Have fun with it**




