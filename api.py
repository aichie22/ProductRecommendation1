# from flask import Flask, jsonify
# from flask_restful import Api, Resource, reqparse
# import pickle
# import numpy as np
# import json

# app = Flask(__name__)
# api = Api(app)

# # Create parser for the payload data
# parser = reqparse.RequestParser()
# parser.add_argument('data')

# # Define how the api will respond to the post requests
# # class IrisClassifier(Resource):
# #     def post(self):
# #         args = parser.parse_args()
# #         X = np.array(json.loads(args['data']))
# #         prediction = model.predict(X)
# #         return jsonify(prediction.tolist())

# # api.add_resource(IrisClassifier, '/iris')

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ProductRecommendation(BaseModel):
    Babies:int
    Children:int
    Adults:int
    Face:int
    Body:int
    Hair:int
    Scalp:int
    Mouth:int
    Psoriasis:int
    Dry_Skin:int
    Normal_Skin:int
    Itchy_Skin:int
    Sensitive_Skin:int
    Inflammed_Skin:int
    Infected_Skin:int
    Dry_Scalp:int
    Itchy_Scalp:int
    Sensitive_Scalp:int
    Oily_Scalp:int
    Red_Scalp:int
    Flaky_Scalp:int
    Wounds:int
    Burns:int
    Blisters:int
    Cuts:int
    Red_Skin:int
    Delicate_Skin:int
    Sun_Rays_Protection:int

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:ProductRecommendation):
    df= pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":int(yhat)}

