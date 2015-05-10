#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script is used to launch marvin.

Application initialization should go here.

'''

import argparse

from flask import Flask

# --------------------------
# Parse command line options
# --------------------------
parser = argparse.ArgumentParser(description='Script to start the SDSS API.')
parser.add_argument('-d','--debug', help='Launch app in debug mode.',action="store_true",required=False)
parser.add_argument('-p','--port', help='Port to use in debug mode.', default=5000,type=int,required=False)
parser.add_argument('-r','--rules', help='List registered rules.',action="store_true",default=False,required=False)

args = parser.parse_args()


# -------------------
# Create app instance
# -------------------
from marvin import create_app

app = create_app(debug=args.debug)

# Can't create the database connection unless we've created the app
from marvin.model.database import db


# ------------------------------------
# register Flask modules (if any) here
# ------------------------------------
#app.register_module(xxx)

# Ref: http://stackoverflow.com/questions/13317536/get-a-list-of-all-routes-defined-in-the-app
# Ref: http://stackoverflow.com/questions/17249953/list-all-available-routes-in-flask-along-with-corresponding-functions-docstrin
if args.rules:
    for rule in app.url_map.iter_rules():
        print("Rule: {0} calls {1} ({2})".format(rule, rule.endpoint, ",".join(rule.methods)))

if __name__ == "__main__":
    '''
    This is called when this script is directly run.
    uWSGI gets the "app" object (the "callable") and runs it itself.
    '''
    if args.debug:
        # Safari blocks some high ports (e.g.port 6000)
        # Ref: http://support.apple.com/kb/TS4639
        app.run(debug=args.debug, port=args.port)
    else:
        app.run()

# PLACE NO CODE BELOW THIS LINE - it won't get called. "app.run" is the main event loop.





