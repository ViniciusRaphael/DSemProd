import pickle
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load(open( r'C:\Users\Vinicius\Desktop\DSemProd\parameter\model_rossman.pkl', 'rb'))

# initialize API
app = Flask( __name__ )

@app.route( '/rossman/predict', methods=['POST'])

def rossman_predict():
    test_json = request.get_json()

    if test_json: # data identified
        if isinstance( test_json, dict):
            # Unique example
            test_raw = pd.DataFrame( test_json, index=[0])
        else:
            # Multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

    # Instantiate Rossman Class
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )

        # feature enginerring
        df2 = pipeline.feature_engineering( df1)
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3)

        return df_response

    else: 
        return Response( '{}', status=200, mimetype='application/json')



if __name__ == '__main__':
    app.run( '192.168.0.7', port=5000, debug=True)