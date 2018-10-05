from functools import wraps
from flask import request, g
import requests
import json
from server import app, logger

from flask_restful import Resource, abort, reqparse


def verify_token(token):
    return token == app.config['SECRET_KEY']

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = request.headers.get('Authorization', None).split('Bearer ')[1]

            if token:
                string_token = token.encode('ascii', 'ignore')
                is_valid = verify_token(string_token)
                logger.debug(is_valid)
                if is_valid:
                    return f(*args, **kwargs)
                else:
                    return abort(401, message="Invalid authentication bearer token")
            else:
                return abort(401, message="Authentication bearer token is required to access this resource")
        except AttributeError as e:
            return abort(401, message="Authentication bearer token is required to access this resource")
        except IndexError as e:
            return abort(401, message="Missing 'Bearer' in authentication")

    return decorated


class AuthenticatedResource(Resource):
    method_decorators = [requires_auth]
