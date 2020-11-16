
The Marvin web server runs off Flask.

# Flask Web Server

Ensure the proper Python packages are installed for web development.  From within the git repo, run:
```
pip install -e .[dev,web]
```

## Local Development

To start the web server locally, in a terminal:

1. Navigate to `bin/` subdirectory in the git repo.
2. Run `run_marvin -d -p [PORT]`.  Specify a local port to run the server on, e.g. 5000.

The server will start.  You should be able to access the web UI at `http://localhost:[PORT]/marvin/`.  The API is accessible at various routes underneath `http://localhost:[PORT]/marvin/api/`.  See the [API docs](https://sdss-marvin.readthedocs.io/en/latest/reference/web.html) for available route documentation.

Changes to any of the Flask views/controllers in `web/controllers` should cause Flask to auto-reload the development server.  The templates are located in `web/templates`.  Javascript code is in `web/lib`.  See [Javascript Development](#Javascript-Development).  Code pertaining to Marvin's programmatic API live in `api/`, along with code in [Marvin's Brain](https://github.com/sdss/marvin_brain/).

## Production Deployment

Production deployment is handled by [nginx](https://nginx.org/en/docs/) and [uwsgi](https://uwsgi-docs.readthedocs.io/en/latest/index.html#).  The uwsgi deployment configuration ini files are located in `web/uwsgi_conf_files`.  Once you have a proper nginx server and location set up, to start a single instance of the web application, run:
```
uwsgi --ini /absolute/path/to/ini/config/file --daemonize /absolute/path/to/logs/file.log
```
The following configurations are available:
- local - a local host
- local_public - a local host of the web server running in "public" mode.
- test - a test server at Utah
- test_public - a test server in "public" mode
- prod -  the production server for the collaboration
- prod_public - the production server in "public" mode.
- jhu - a server for the mirror at JHU.

The server runs using a uwsgi [master FIFO](https://uwsgi-docs.readthedocs.io/en/latest/MasterFIFO.html), rather than pid files.  For example, to gracefully reload the application, run `echo r > marvin_fifo`.


# Javascript Development

Marvin javascript code is currently included in html as scripts and stylesheets.  All related javascript code and css is contained in the `web/lib` subdirectory.  To it up for develpment, perform the following:

1. Navigate to the `lib` directory.
2. Run `npm install` to install the node packages locally.  If you don't have `npm`, get it with [node](https://nodejs.org/en/).

The javacsript source code is in the `lib/src` directory.  The SASS (scss) source code is in the `lib/scss` directory.  The source for static images is in `lib/images`.  Once changes are made to files in these directories, you must recompile the javascript and css.

From the `lib` directory, run `grunt -v --trace`.  This runs the [Grunt](https://gruntjs.com/) task manager.  Grunt runs the following tasks:

1. Transpiles the javascript into ES2015, located in `lib/js`.
2. Compiles the SASS into CSS styling, located in `lib/css`.
3. Combines all the source javascript and css files together into single files located in `lib/dist`
4. Performs image, css, and js minification.  The images files are located in `static/`.  The js and css files are located in `lib/dist`.

The `marvin.min.js` and `marvin.min.css` files located in `lib/dist` are what is included in the html files.  The `modernizer-custom.js` is needed to check browser support for the `webp` image format.

# Bootstrap Development

The Marvin web UI uses Bootstrap for all SASS and CSS stylings.  Currently the web layout in `web/templates` uses [Bootstrap 3.3.7](https://getbootstrap.com/docs/3.3/).  Additionally bootstrap plugins used are:

- Bootstrap Table 1.16.0
- Bootstrap-Select 1.10.0
- Bootstrap-Toggle 2.2.2
- Bootstrap-3-Typeahead 4.0.1

## Additional Javascript Libraries
- Jquery 3.2.1
- CookieConsent 3.0.3
- D3 v4
- Dygraphs 2.0.1
- Highcharts 8.2.2
- Open Layers 3.14.2
- Query Builder 2.4.3
- Raven 3.7.0


# Testing

Tests for the web server are located in `tests/web` from the top level of the git repo.  You can run them with pytest.  Most tests here are for the API and are headless.  Front-end tests for the user interfaces are located in `tests/web/frontend` and are run using Selenium and pytest-flask.