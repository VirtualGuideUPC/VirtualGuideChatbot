from flask import Flask, request, session
from flask_restful import Api, Resource
from flask import jsonify
from flask_restful.utils import cors
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
        if "place_context" in session:
            place_context = session["place_context"]
            print(place_context)
        else:
            place_context = " "

        msg = request.json['msg']
        cb = ChatBot(msg, place_context)
        cb.create_response()
        cb.select_response()
        message = cb.res

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

        response = {
            'msg': message
        }

        return jsonify(to_dict(response))


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
