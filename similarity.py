from urllib.parse import unquote
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import requests
import itertools
import pathlib
import re
import sys
import json


ABSTRACT_URL_ENDPOINT_WEIGHT = 2
TOKEN_PATTERN = r"\b[a-zA-Z0-9_-]+\b"
TOKEN_RE = re.compile(TOKEN_PATTERN)


def query_duckduckgo(query, no_html=1, skip_disambig=1, **kwargs):
    params = {
        "q": query,
        "no_redirect": 1,
        "no_html": no_html,
        "skip_disambig": skip_disambig
    }
    params.update(kwargs)
    url = 'https://api.duckduckgo.com/'
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None


def regularize_name(name):
    """Remove all special character and extra whitespace in a name
    First tokenize it, convert to lower case, then chain the tokens by a single space
    """
    return " ".join([t.lower() for t in re.findall(TOKEN_RE, name)])


def get_corpus_for_names(names, abstract_url_weight):
    text_corpus = []
    seen_abstract = set()

    answers_from_duckduckgo = list(map(lambda x: query_duckduckgo(x, format="json"), names))
    for answer in answers_from_duckduckgo:
        abstract = answer["Abstract"]
        abstract_url = answer["AbstractURL"]

        if abstract and abstract not in seen_abstract:
            #rule out duplicate abstract
            seen_abstract.add(abstract)

            if abstract_url:
                abstract_url = unquote(abstract_url)
                url_endpoint = abstract_url.split("/")[-1]
                url_endpoint = url_endpoint.replace("_", " ")
                weighted_url_endpoint = [url_endpoint]*abstract_url_weight
                abstract += " ".join(weighted_url_endpoint)

            text_corpus.append(abstract)

    return text_corpus


def get_tfidf(text_corpus):
    vectorizer  = CountVectorizer(strip_accents="unicode", analyzer="word",
            token_pattern = TOKEN_PATTERN, lowercase=True)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text_corpus))
    return vectorizer.vocabulary_, tfidf



def get_similarity_features(names, text_corpus_path):
    #avoid download again and again
    if not pathlib.Path(text_corpus_path).is_file():
        text_corpus = get_corpus_for_names(names, ABSTRACT_URL_ENDPOINT_WEIGHT)
        with open(text_corpus_path, "w") as f:
            for text in text_corpus:
                f.write(text + "\n")
    else:
        text_corpus = []
        with open(text_corpus_path, "r") as f:
            for line in f:
                text_corpus.append(line)

    word_idx_map, tfidf = get_tfidf(text_corpus)

    feature_dim = tfidf.shape[0]

    def get_features_for_a_single_name(name):
        feature_vec = np.zeros(feature_dim)
        tokens = map(lambda x: x.lower(), re.findall(TOKEN_PATTERN, name))
        connected_fullname = ""

        #add all the tfidf vec of its words of a name
        for word in tokens:
            word = word.lower()
            connected_fullname += word
            idx  = word_idx_map.get(word, None)
            if idx:
                feature_vec += tfidf[:, idx].toarray()[:, 0]

        #case the name is fully connected in corpus, ex: joeystarr
        fullname_idx = word_idx_map.get(connected_fullname, None)
        if fullname_idx:
            feature_vec += tfidf[:, fullname_idx].toarray()[:, 0]

        return feature_vec

    features     = [get_features_for_a_single_name(name) for name in names]
    feature_cols = ["feature_%d" % c for c in range(0, feature_dim)]
    feature_df   = pd.DataFrame.from_records(features, index=names, columns=feature_cols)
    return feature_df


def get_similarity_scores(feature_df, pseudonyms, realnames):
    score_matrix = np.array([np.dot(np.array(feature_df.loc[x]), np.array(feature_df.loc[y]))
            for x, y in itertools.product(pseudonyms, realnames)])
    score_matrix = score_matrix.reshape(len(pseudonyms), len(realnames))
    score_df = pd.DataFrame(score_matrix, index=pseudonyms, columns=realnames)
    #find the best match
    #score_df["match"] = [score_df.loc[name].idxmax() for name in score_df.index]
    return score_df


if __name__ == '__main__':
    pseudonyms_path = sys.argv[1]
    realnames_path  = sys.argv[2]

    pseudonyms_df = pd.read_csv(pseudonyms_path, names=["pseudonym"])
    realnames_df  = pd.read_csv(realnames_path, names=["realname"])

    pseudonyms = [regularize_name(name) for name in pseudonyms_df["pseudonym"]]
    realnames  = [regularize_name(name) for name in realnames_df["realname"]]

    #normalize pseudonyms
    all_names = list(itertools.chain(pseudonyms, realnames))

    feature_df = get_similarity_features(all_names, "corpus.txt")
    feature_df.to_csv("features.csv")

    score_df = get_similarity_scores(feature_df, pseudonyms, realnames)
    score_df.to_csv("scores.csv")
