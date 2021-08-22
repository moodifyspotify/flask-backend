from flask import Flask, jsonify, request,render_template
from flask_cors import CORS, cross_origin
import pandas as pd
import json
import plotly
import plotly.express as px
from get_songs_data import SongProcessing
import json



def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    CORS(app, resources={r"/api/*": {"origins": "*"}})


    @app.route('/')
    def main_page():
        return "heh"

    @app.route('/get_songs_history')
    def songs_history():
        sngs = SongProcessing('AQAAAAAkI_QrAAG8Xhhgt83_bk-OlbRo6xG86wM')
        return json.dumps(sngs.get_user_songs_history())
        # return str(sngs.ya_client.me.account.uid)

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
