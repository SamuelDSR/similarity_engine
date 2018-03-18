from urllib.parse import unquote
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

from unidecode import unidecode
import pandas as pd
import numpy as np
import requests
import itertools
import pathlib
import re
import sys
import json


ABSTRACT_URL_ENDPOINT_WEIGHT = 2
TOKEN_PATTERN = r"\b[a-zA-Z0-9]+\b"
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


def regularize_string(string):
    """strip accents, extract all word tokens, convert it to lower case,
    return the transformed tokens chained by a single space
    """
    string = unidecode(string)
    return " ".join([t.lower() for t in re.findall(TOKEN_RE, string)])


def get_heading_feature(answers_from_duckduckgo):
    headings = [regularize_string(answer["Heading"])
            for answer in answers_from_duckduckgo]
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(list(filter(lambda x: x, headings)))
    return label_binarizer.transform(headings)


def get_abstract_url_feature(answers_from_duckduckgo):
    url_endpoints = [unquote(answer["AbstractURL"])
            for answer in answers_from_duckduckgo]
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(list(filter(lambda x: x, url_endpoints)))
    return label_binarizer.transform(url_endpoints)


def get_image_url_feature(answers_from_duckduckgo):
    image_urls = [unquote(answer["Image"]) for answer in answers_from_duckduckgo]
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(list(filter(lambda x: x, image_urls)))
    return label_binarizer.transform(image_urls)


def process_text_features(names, text_corpus):
    vectorizer  = CountVectorizer(strip_accents="unicode", analyzer="word",
            token_pattern = TOKEN_PATTERN, lowercase=True)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text_corpus))

    feature_dim      = tfidf.shape[0]
    vocalbulary_size = tfidf.shape[1]

    vocalbulary_map  = vectorizer.vocabulary_

    def get_coexistence_feature_for_a_single_name(names):
        feature_vec = np.zeros((1,feature_dim))
        tokens = names.split(" ")
        for token in tokens:
            idx = vocalbulary_map.get(token, None)
            if idx is not None:
                feature_vec += tfidf[:, idx].toarray().reshape(1, feature_dim)
        return feature_vec

    features = []
    for name in names:
        features.append(get_coexistence_feature_for_a_single_name(name))

    return np.array(features).reshape(len(names), feature_dim)

def get_abstract_feature(names, answers_from_duckduckgo):
    abstracts = [answer["Abstract"] for answer in answers_from_duckduckgo]
    return process_text_features(names, abstracts)


def get_born_or_birth_name_feature(names, answers_from_duckduckgo):
    def get_infobox_key_info(infobox, key):
        if isinstance(infobox, str):
            return infobox
        else:
            values_list = infobox.get("content", None)
            if values_list:
                for value in values_list:
                    if value["label"] == key:
                        return value["value"]
            return ""

    names_regex = r"(" + "|".join(names) + r")"
    names_pat = re.compile(names_regex)
    names_map = dict([(name, idx) for idx, name in enumerate(names)])

    born_or_birth_name_features = np.identity(len(names))

    for name, answer in zip(names, answers_from_duckduckgo):
        born = regularize_string(get_infobox_key_info(answer["Infobox"], "Born"))
        born_names = names_pat.findall(born)
        if len(born_names) != 0:
            born_or_birth_name_features[names_map[name], names_map[born_names[0]]] += 1  
            
        birth_name  = regularize_string(get_infobox_key_info(answer["Infobox"], "Birth name"))
        birth_names = names_pat.findall(birth_name)
        if len(birth_names) != 0:
            born_or_birth_name_features[names_map[name], names_map[birth_names[0]]] += 1

    return born_or_birth_name_features



def get_similarity_features(names, answers_from_duckduckgo):
    heading_feature = get_heading_feature(answers_from_duckduckgo)
    abstract_url_feature = get_abstract_url_feature(answers_from_duckduckgo)
    image_url_feature = get_image_url_feature(answers_from_duckduckgo)
    abstract_feature = get_abstract_feature(names, answers_from_duckduckgo)
    born_or_birth_name_feature = get_born_or_birth_name_feature(names, answers_from_duckduckgo)

    print(heading_feature.shape)
    print(abstract_url_feature.shape)
    print(image_url_feature.shape)
    print(abstract_feature.shape)
    print(born_or_birth_name_feature.shape)

    features = np.concatenate((heading_feature, abstract_url_feature,
        image_url_feature, abstract_feature, born_or_birth_name_feature), axis=1)
    features_df = pd.DataFrame.from_records(features, index=names)
    return features_df


def get_similarity_scores(feature_df, pseudonyms, realnames):
    score_matrix = np.array([np.dot(np.array(feature_df.loc[x]), np.array(feature_df.loc[y]))
            for x, y in itertools.product(pseudonyms, realnames)])
    score_matrix = score_matrix.reshape(len(pseudonyms), len(realnames))
    score_df = pd.DataFrame(score_matrix, index=pseudonyms, columns=realnames)
    #find the best match
    score_df["match"] = [score_df.loc[name].idxmax() for name in score_df.index]
    return score_df


if __name__ == '__main__':
    pseudonyms_path = sys.argv[1]
    realnames_path  = sys.argv[2]

    pseudonyms_df = pd.read_csv(pseudonyms_path, names=["pseudonym"])
    realnames_df  = pd.read_csv(realnames_path, names=["realname"])

    pseudonyms = [regularize_string(name) for name in pseudonyms_df["pseudonym"]]
    realnames  = [regularize_string(name) for name in realnames_df["realname"]]

    #regularize pseudonyms
    all_names = list(itertools.chain(pseudonyms, realnames))
    
    #query answers from duckduckgo
    answers_from_duckduckgo = [query_duckduckgo(name, format="json")
            for name in all_names]
    with open("answers.json", "w") as f:
        json.dump(answers_from_duckduckgo, f)

    #with open("answers.json", "r") as f:
        #answers_from_duckduckgo = json.load(f)


    features_df = get_similarity_features(all_names, answers_from_duckduckgo)
    features_df.to_csv("features.csv")

    score_df = get_similarity_scores(features_df, pseudonyms, realnames)
    score_df.to_csv("scores.csv")
