#!/usr/bin/env python

'''

This script is used to profile the mangasas web-app.  It runs the app in debug mode, 
on default port 5000.

Type ./profile_mangasas.py 

Navigate the site as usual.  All requests get profiled and output in the terminal screen.   

'''

from flask import Flask
from werkzeug.contrib.profiler import ProfilerMiddleware
from mangasas import create_app

app = create_app(debug=True)

from mangasas.model.database import db


app.config['PROFILE'] = True
app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
app.run(debug = True)



