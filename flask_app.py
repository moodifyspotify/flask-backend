from flask import Flask, jsonify, request,render_template
from flask_cors import CORS, cross_origin
import json

import random

import json
from datetime import datetime
import time
import ssl
import pymongo
from urllib.parse import quote_plus as quote
from bson.objectid import ObjectId
import base64
import secrets
import pandas as pd
import json
import plotly
import plotly.express as px



def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    CORS(app, resources={r"/api/*": {"origins": "*"}})


    @app.route('/')
    def main_page():
        return 'Tell me you are sad without telling me you are sad'

    @app.route('/dash_test',methods=['GET', 'POST'])
    def notdash():
        df = pd.DataFrame({
            "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
            "Amount": [4, 1, 2, 2, 4, 5],
            "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
        })
        fig = px.bar(df, x="Fruit", y="Amount", color="City",    barmode="group")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('notdash.html', graphJSON=graphJSON)


    @app.route('/api', methods=['GET'])
    @cross_origin()
    def default_f():
        if request.method == "GET":
            return {'result':'It Works!'}, 200
        return jsonify({
                    'statusCode': 400
                }), 400


    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0',debug=True)

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.express as px
# import pandas as pd
# app = dash.Dash(__name__)
# df = pd.DataFrame({
#    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#    "Amount": [4, 1, 2, 2, 4, 5],
#    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
# fig = px.bar(df, x="Fruit", y="Amount", color="City",  
#    barmode="group")
# app.layout = html.Div(children=[
#    html.H1(children="Hello Dash"),
#    html.Div(children="""
#    Dash: A web application framework for Python.
#    """),
#    dcc.Graph(
#       id="example-graph",
#       figure=fig
#    )
# ]) 
# if __name__ == "__main__":
#    app.run_server(debug=True)
