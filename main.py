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
    @cors.crossdomain(origin='*', methods={"HEAD", "OPTIONS", "GET", "POST"})
    
    def post(self):
        msg = request.json['text']
        user = request.json['user']
        try:
            url = request.json['url']
        except:
            url=""
            
        cb = ChatBot(msg, user)
        many_candidates = cb.set_message()
        
        if many_candidates:
            index=0
            cb.selec_from_candidates(index)
        
        cb.confirm_candidate()
        cb.save_context(user)
        cb.create_response(user)
        cb.select_response()            
        message=cb.res
        if cb.show_image:
            botUrl = cb.get_url_image()
        else: botUrl = ""
                
        url="http://ec2-52-90-137-95.compute-1.amazonaws.com/api/users/message/create/"
        payload=json.dumps(request.json)
        headers={
            'Content-Type': 'application/json'
        }
        apiresponse=requests.request("POST",url,headers=headers, data=payload)
        
        if apiresponse.ok:
            request.json['text']= message
            request.json['is_user'] = False
            request.json['url'] = botUrl
            
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
