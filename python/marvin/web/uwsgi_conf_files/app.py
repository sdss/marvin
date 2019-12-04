# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Filename: app.py
# Project: uwsgi_conf_files
# Author: Brian Cherinka
# Created: Tuesday, 3rd December 2019 9:47:55 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2019 Brian Cherinka
# Last Modified: Tuesday, 3rd December 2019 9:48:59 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import
from marvin.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
