from flask import Flask, request, session
from flask_restful import Api, Resource
from flask import jsonify
from flask_restful.utils import cors
import requests
import json
from flask_cors import CORS
from chatbot import ChatBot
from datetime import timedelta

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)
app.secret_key = "vguidesecurity"
app.permanent_session_lifetime = timedelta(minutes=5)

def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


class Init(Resource):
    @cors.crossdomain(origin='*',
                      methods={"HEAD", "OPTIONS", "GET", "POST"})
    def get(self):
        response = {
            'msg': "Hola! Soy A.V.T... El Asistente Virtual de Turismo, dime, ¿qué puedo hacer por ti?"
        }
        return jsonify(response)


class Prediction(Resource):
    @cors.crossdomain(origin='*',
                      methods={"HEAD", "OPTIONS", "GET", "POST"})
    def post(self):

        # validate if exist place context cookie
        if "place_context" in session:
            place_context = session["place_context"]
            print(place_context)
        else:
            place_context = " "

        # changed 'msg' to 'text' according to rest api model
        msg = request.json['text']
        cb = ChatBot(msg, place_context)

        if "places_candidates" in session:
            place_candidates = session["places_candidates"]
            cb.intencion = session["intention"]
            cb.isPlaces = True
        else:
            place_candidates = cb.set_message()
            if len(place_candidates) > 1:
                session.permanent = True
                session["places_candidates"] = place_candidates
                session["intention"] = cb.intencion
            if cb.isPlaces:
                session.pop("places_candidates", None)
                session.pop("intention", None)

        cb.select_candidate(place_candidates)

        if cb.isPlacesSelected:
            session.pop("places_candidates", None)
            session.pop("intention", None)

        if not cb.isPlaces:
            cb.create_response()
            cb.select_response()

        message = cb.res

        # Assign value for place context cookie
        if "place_context" in session:
            if cb.place_context == " ":
                place_context = session["place_context"]
                print(place_context)
            else:
                place_context = cb.place_context
                session.permanent = True
                session["place_context"] = place_context
        else:
            place_context = cb.place_context
            session.permanent = True
            session["place_context"] = place_context

        print("context:")
        print(place_context)

        #Post to API
        url="http://ec2-34-226-195-132.compute-1.amazonaws.com/api/users/message/create/"
        payload=json.dumps(request.json)
        headers={
            'Content-Type': 'application/json'
        }
        apiresponse=requests.request("POST",url,headers=headers, data=payload)
        if apiresponse.ok:
            request.json['text']=message
            request.json['is_user']=False
            payloadbot=json.dumps(request.json)
            apiresponsebot=requests.request("POST",url,headers=headers,data=payloadbot)
            print(apiresponse.json())
            if apiresponsebot.ok:
                response={
                    'human_message':apiresponse.json(),
                    'robot_response':apiresponsebot.json()
                }
                print(apiresponsebot.json())
                return jsonify(to_dict(response))
            else:
                response={
                    'human_message':apiresponse.json(),
                    'robot_response': 'error posting response to db'
                }
                return jsonify(to_dict(response))
        else:
            response={
                'human_message':'error posting message to db'
            }
            return jsonify(to_dict(response))
        
        
        


        # response = {
        #     'msg': message
        # }

        # print(request.json)
        # return jsonify(to_dict(response))


class PopCookie(Resource):
    @cors.crossdomain(origin='*',
                      methods={"HEAD", "OPTIONS", "GET", "POST"})
    def delete(self):
        session.pop("place_context", None)
        return jsonify("cookie delete")


api.add_resource(Init, '/init')
api.add_resource(Prediction, '/prediction')
api.add_resource(PopCookie, '/popcookie')

if __name__ == '__main__':
    app.run(debug=True)
