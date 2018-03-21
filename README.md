## Similarity engine

A simple similarity engine that can match the pseudonym and the real name of an artist.

It use the answer returned from [duckduckgo](https://duckduckgo.com/api) to build several
features.
The main feature is based on the **tfidf** of the abstracts of *real names* and *pseudonym*, the dot product between the features denotes the similiarites.

You could find detailed explanation of how these features are built in the attached *jupyter notebook*.


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




