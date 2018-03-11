import os
import json

import dash
import flask
from flask import request

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from similarity import regularize_name

features_df = pd.read_csv("features.csv", index_col=0)
scores_df   = pd.read_csv("scores.csv", index_col=0)

site_dir = os.getcwd()
allowed_stylesheets = ["style.css"]

app = dash.Dash()


@app.server.route("/static/<stylesheet>")
def serve_stylesheet(stylesheet):
    if stylesheet not in allowed_stylesheets:
        raise Exception(
                '"{}" is not allowed on this site'.format(stylesheet)
                )
    return flask.send_from_directory(site_dir, stylesheet)


@app.server.route("/similarity", methods=["Get"])
def api_get_similarity():
    pseudonym  = regularize_name(request.args.get("pseudonym", ""))
    real_name  = regularize_name(request.args.get("real_name", ""))

    similarity = {}

    unseen_words = []
    if pseudonym not in features_df.index:
        unseen_words.append(pseudonym)
        similarity[pseudonym] = None
    else:
        similarity[pseudonym] = features_df.loc[pseudonym].tolist()

    if real_name not in features_df.index:
        unseen_words.append(real_name)
        similarity[real_name] = None
    else:
        similarity[real_name] = features_df.loc[real_name].tolist()

    if unseen_words:
        similarity["score"] = None
    else:
        similarity["score"] = scores_df.at[pseudonym, real_name]

    return json.dumps(similarity)

app.css.append_css({"external_url":"/static/style.css"})

app.layout = html.Div([
        html.H2(children="Find similarity between pseudonym and real name of an artist"),
        html.Div([
            html.Div([
                html.Label(children="Pseudonym"),
                dcc.Input(id="input_pseudo",
                    type="text", placeholder="Enter the pseudonym of artist")
                ], className="inputdivcls"),
            html.Div([
                html.Label(children="Real Name"),
                dcc.Input(id="input_real", 
                    type="text", placeholder="Enter the real name of the artist")
                ], className="inputdivcls"),
            html.Div([
                html.Button("Submit", id="submit")
                ])
            ]),
        html.Div([
            dcc.Markdown(id="output_text")
            ], id="outputdiv")
        ])

markdown_output_format = """
The feature vector of **{pseudonym}** is: {pseudonym_feature}.

The feature vector of **{real_name}** is: {real_name_feature}.

Similarity between **{pseudonym}** and **{real_name}** is : **{score}**.
"""

error_output_format = """
Sorry, the input {} is not found in our database.
"""
@app.callback(
        Output("output_text", "children"),
        [Input("submit", "n_clicks")],
        [State("input_pseudo", "value"), State("input_real", "value")])
def dash_get_similarity(n_clicks, pseudonym, real_name):
    pseudonym = regularize_name(pseudonym)
    real_name = regularize_name(real_name)
    unseen_words = []
    if pseudonym not in features_df.index:
        unseen_words.append(pseudonym)
    if real_name not in features_df.index:
        unseen_words.append(real_name)

    if unseen_words:
        return error_output_format.format(unseen_words)

    pseudonym_feature = features_df.loc[pseudonym].tolist()
    real_name_feature = features_df.loc[real_name].tolist()
    score = scores_df.at[pseudonym, real_name]

    return markdown_output_format.format(
            pseudonym=pseudonym, real_name=real_name,
            pseudonym_feature=pseudonym_feature,
            real_name_feature=real_name_feature,
            score=score
            )


if __name__ == "__main__":
    app.run_server(host="0.0.0.0")
