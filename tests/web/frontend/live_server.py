# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-06 22:36:32
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-02-14 15:18:26

from __future__ import print_function, division, absolute_import
import multiprocessing
import time
import requests
import pytest


class LiveServer(object):
    """The helper class uses to manage live server. Handles creation and
    stopping application in a separate process.
    :param app: The application to run.
    :param port: The port to run application.
    """

    def __init__(self, app, port):
        self.app = app
        self.port = port
        self._process = None

    def start(self):
        """Start application in a separate process."""
        def worker(app, port):
            app.run(port=port, use_reloader=False, threaded=True)
        self._process = multiprocessing.Process(
            target=worker,
            args=(self.app, self.port)
        )
        self._process.start()

        # We must wait for the server to start listening with a maximum
        # timeout of 5 seconds.
        timeout = 5
        while timeout > 0:
            time.sleep(1)
            try:
                requests.get(self.url())
                timeout = 0
            except:
                timeout -= 1

    def url(self, url=''):
        """Returns the complete url based on server options."""
        return 'http://localhost:%d%s' % (self.port, url)

    def stop(self):
        """Stop application process."""
        if self._process:
            self._process.terminate()

    def __repr__(self):
        return '<LiveServer listening at %s>' % self.url()


@pytest.fixture(scope='session')
def live_server(request, app):
    """Run application in a separate process.
    When the ``live_server`` fixture is applyed, the ``url_for`` function
    works as expected::
        def test_server_is_up_and_running(live_server):
            index_url = url_for('index', _external=True)
            assert index_url == 'http://localhost:5000/'
            res = urllib2.urlopen(index_url)
            assert res.code == 200
    """
    # Bind to an open port
    port = app.config.get('LIVESERVER_PORT', 5000)

    # Explicitly set application ``SERVER_NAME`` for test suite
    # and restore original value on test teardown.
    server_name = app.config['SERVER_NAME'] or 'localhost'
    app.config['SERVER_NAME'] = '{0}:{1}'.format(server_name, port)

    server = LiveServer(app, port)
    if request.config.getvalue('start_live_server'):
        server.start()

    yield server

    # teardown
    server.stop()
