from flask import Flask
from .main import main

def create_app(args, imt_system, testset):
    app = Flask(__name__)
    if not hasattr(app, 'extensions'):
        app.extensions = {}
    app.extensions["args"] = args
    app.extensions["imt_system"] = imt_system
    app.extensions["testset"] = testset
    app.register_blueprint(main)

    return app