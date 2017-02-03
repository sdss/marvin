
.. _marvin-api:

API
===

API stands for Application Programming Interface.  It describes a set of rules designed to faciliate remote acquisition of data, without using a user interface.  It is typically designed as a set of HTTP Request methods (i.e. GET or POST), that you can interact with in a browser, or via code packages, e.g. the Python package `Requests <http://docs.python-requests.org/en/master/>`_.  These URL routes, along with all their parameters, can sometimes be tedious to deal with explicitly.

Marvin makes this easy for you in two ways:

1. The Marvin API wraps this functionality into an :ref:`marvin-interaction-class`, making it easy for you to make these calls if you want.
2. All Marvin Tools have an API call built in when interacting in 'remote' mode, that uses the :ref:`marvin-interaction-class` already, so you don't have to.

With the Marvin API, you can build your own application, website, or scripts very easily, and forego the rest of Marvin, if so desired.  Currently all API requests will timeout after 5 minutes.  See :ref:`marvin-api-routes` for a list of the routes available in the Marvin API.

.. _marvin-urlmap:

Config.Urlmap
-------------

The Marvin.config.urlmap is a nested lookup dictionary that contains all of the API routes used in Marvin.  If you have a connection
to the internet, upon intial import, Marvin will attempt to build the urlmap by contacting Marvin at Utah.  With a valid
internet connection, and config.sasurl variable, Marvin will populate the urlmap with all of the API routes available to use.

The API routes are contained in a key called **api**.  The list of available API endpoints are available as dictionary keys, with the urls for each endpoint available in the dictionary key **url**.

Urlmap Syntax: config.urlmap[**page**][**endpoint**][**url**]

* **page**: The specific page you want to look at.  For the API, this key is **api**.
* **endpoint**: A shortcut name pointing to the URL route defined on that method.
* **url**: The string url path needed to pass into the Marvin Interaction class

Usage
::

    from marvin import config

    # look at available urlmap API endpoints
    print(config.urlmap['api'].keys())
    [u'getroutemap', u'querycubes', u'getCube', u'getspectra', u'getparams', u'getspaxels', u'getSpaxel', u'mangaid2plateifu', u'getRSS', u'getPlate', u'getPlateCubes', u'webtable']

    # get the URL for getting basic Cube properties
    url = config.urlmap['api']['getCube']['url']

    # print(url)
    u'/marvin2/api/cubes/{name}/'

Some urls require parameters passed to them. Others do not.  Urls with curly braces {} in them indicate an input parameter. For example, in the above url, ```{name}``` means a parameter called name must be passed into the url. See how to pass in parameters in the examples below.

.. _marvin-authentication:

API Authentication
------------------

The use of the API requires authentication.  To authenticate, you will need to have a .netrc file in your local home directory.  Inside the .netrc file,
::

    # create a .netrc file if you do not already have one
    cd ~/
    touch .netrc

    # using a text editor, place the following text inside your .netrc file.
    machine api.sdss.org
        login sdss
        password replace_with_sdss_password

.. _marvin-interaction-class:

Interaction Class
-----------------

If you want to explicitly grab the data remotely outside of Marvin Tools, you can do so with the :ref:`marvin-api-interaction` class. This class, in combination with the Marvin :ref:`marvin-urlmap`, allows you to easily make API requests and retrieve the results.  The Interaction class returns data in a basic JSON format and translates it to a more user-friendly python data types (e.g. numpy arrays).


Usage:
::

    from marvin import config
    config.mode = 'remote'

    # import the Marvin Interaction class
    from marvin.api.api import Interaction

    # get and format an API url to retrieve basic Cube properties
    plateifu = '7443-12701'
    url = config.urlmap['api']['getCube']['url']

    # create and send the request, and retrieve a response
    response = Interaction(url.format(name=plateifu))

    # check your response's status code
    print(response.status_code)
    200

    # get the data in your response
    data = response.getData()
    print(data)


Http Status Codes
-----------------
These tell you whether or not your request was successful.  A status code of 200 mean success.  Any other status code means failure.  If the Interaction requset fails, you will receive a dictionary containing the status code, and an error message.

Status Codes:

* **200**: OK
* **404**: Page Not Found - the page connected to the input route does not exist
* **500**: Internal Server Error - something has gone wrong on the server side
* **405**: Method Not Allowed - the route is using the wrong method request, e.g. GET instead of POST
* **401**: Authentication Required - the correct authentication credentials was not provided
* **422**: Unprocessable Entity - the input parameters are invalid
* **400**: Bad Request
* **502**: Bad Gateway
* **504**: Gateway Timeout


