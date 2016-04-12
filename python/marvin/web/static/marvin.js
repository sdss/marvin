/*
* @Author: Brian Cherinka
* @Date:   2016-04-10 11:42:17
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-11 16:04:40
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//

'use strict';

var Galaxy,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Galaxy = (function () {

    marvin.Galaxy = Galaxy;

    // Constructor
    function Galaxy(plateifu) {

        // in case constructor called without new
        if (false === (this instanceof Galaxy)) {
            return new Galaxy();
        }

        this.init(plateifu);

        // Event Handlers
    }

    // initialize the object
    Galaxy.prototype.init = function init(plateifu) {
        this.setPlateIfu(plateifu);
        this.maindiv = $('#'+this.plateifu);
        this.mapdiv = this.maindiv.find('#map');
        this.specdiv = this.maindiv.find('#graphdiv');
        this.specmsg = this.maindiv.find('#specmsg');
        this.webspec = null;
        this.staticdiv = this.maindiv.find('#staticdiv');
        this.dynamicdiv = this.maindiv.find('#dynamicdiv');
        this.togglediv = $('#toggleinteract');
    };

    // test print
    Galaxy.prototype.print = function print() {
        console.log('We are now printing galaxy', this.plateifu, this.plate, this.ifu);
    };

    // Determine and Set the plateifu from input
    Galaxy.prototype.setPlateIfu = function setPlateIfu(plateifu) {
        if (plateifu === undefined) {
            this.plateifu = $('.galinfo').attr('id');
        } else {
            this.plateifu = plateifu;
        }
        [this.plate, this.ifu] = this.plateifu.split('-');
    };

    // Initialize and Load a DyGraph spectrum
    Galaxy.prototype.loadSpaxel = function loadSpaxel(spaxel) {
        this.webspec = new Dygraph(this.specdiv[0],
                  spaxel,
                  {
                    labels: ['x','Flux'],
                    errorBars: true
                  });
    };

    // Update a DyGraph spectrum
    Galaxy.prototype.updateSpaxel = function updateSpaxel(spaxel, specmsg) {
        var newmsg = "Here's a spectrum: "+specmsg;
        this.specmsg.empty();
        this.specmsg.html(newmsg);
        this.webspec.updateOptions({'file': spaxel});
    };

    // Initialize OpenLayers Map
    Galaxy.prototype.initOpenLayers = function initOpenLayers(image) {
        this.image = image;
        this.olmap = new OLMap(image);
        // add click event handler on map to get spaxel
        this.olmap.map.on('singleclick', this.getSpaxel, this);
    };

    // Retrieves a new Spaxel from the server based on a given mouse position
    Galaxy.prototype.getSpaxel = function getSpaxel(event) {
        var map = event.map;
        var mousecoords = event.coordinate;
        var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords'];
        var form = utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords);
        var _this = this;

        // send the form data
        $.post(Flask.url_for('galaxy_page.getspaxel'), form,'json')
            .done(function(data) {
                $('#mouse-output').empty()
                var myhtml = "<h5>My mouse coords "+mousecoords+", message: "+data.result.message+"</h5>"
                $('#mouse-output').html(myhtml);
                _this.updateSpaxel(data.result.spectra, data.result.specmsg);
            })
            .fail(function(data) {
                $('#mouse-output').empty()
                var myhtml = "<h5>Error message: "+data.result.message+"</h5>"
                $('#mouse-output').html(myhtml);
            });
    };

    // Toggle the interactive OpenLayers map and Dygraph spectra
    Galaxy.prototype.toggleInteract = function toggleInteract(spaxel, image) {
        if (this.togglediv.hasClass('active')){
            this.togglediv.button('reset');
            this.dynamicdiv.hide();
            this.staticdiv.show();
        } else {
            this.togglediv.button('complete');
            this.staticdiv.hide();
            this.dynamicdiv.show();

            // check for empty divs
            var specempty = this.specdiv.is(':empty');
            var mapempty = this.mapdiv.is(':empty');
            // load the spaxel if the div is initially empty;
            if (this.specdiv !== undefined && specempty) {
                this.loadSpaxel(spaxel);
            }

            // load the map if div is empty
            if (mapempty) {
                this.initOpenLayers(image);
            }

        }
    };

    return Galaxy;

})();


;/*
* @Author: Brian Cherinka
* @Date:   2016-04-11 10:38:25
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-11 11:43:09
*/

//
// Javascript object handling all things related to OpenLayers Map
//

'use strict';

var OLMap,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

OLMap = (function () {

    marvin.OLMap = OLMap;

    // Constructor
    function OLMap(image, requestedOptions) {

        // in case constructor called without new
        if (false === (this instanceof OLMap)) {
            return new OLMap();
        }

        this.init(image);

        // Event Handlers
    }

    // initialize the object
    OLMap.prototype.init = function(image) {
        if (image === undefined) {
            console.error('Must specify an input image to initialize a Map!');
        } else {
            this.image = image;
            this.staticimdiv = $('#staticimage')[0];
            this.mapdiv = $('#map')[0];
            this.getImageSize();
            this.setProjection();
            this.setView();
            this.initMap();
            this.addDrawInteraction();
        }

    };

    // test print
    OLMap.prototype.print = function() {
        console.log('We are now printing openlayers map');
    };

    // Get the natural size of the input static image
    OLMap.prototype.getImageSize = function() {
        if (this.staticimdiv !== undefined) {
            this.imwidth = this.staticimdiv.naturalWidth;
            this.imheight = this.staticimdiv.naturalHeight;
        }
    };

    // Set the mouse position control
    OLMap.prototype.setMouseControl = function() {
        var mousePositionControl = new ol.control.MousePosition({
            coordinateFormat: ol.coordinate.createStringXY(4),
            projection: 'EPSG:4326',
            // comment the following two lines to have the mouse position be placed within the map.
            //className: 'custom-mouse-position',
            //target: document.getElementById('mouse-position'),
            undefinedHTML: '&nbsp;'
        });
        return mousePositionControl;
    };

    // Set the image Projection
    OLMap.prototype.setProjection = function() {
      this.extent = [0, 0, this.imwidth, this.imheight];
      this.projection = new ol.proj.Projection({
        code: 'ifu',
        units: 'pixels',
        extent: this.extent
      });
    };

    // Set the base image Layer
    OLMap.prototype.setBaseImageLayer = function() {
        var imagelayer = new ol.layer.Image({
            source: new ol.source.ImageStatic({
                url: this.image,
                projection: this.projection,
                imageExtent: this.extent
                })
        })
        return imagelayer;
    };

    // Set the image View
    OLMap.prototype.setView = function() {
        this.view = new ol.View({
            projection: this.projection,
            center: ol.extent.getCenter(this.extent),
            zoom: 0,
            maxZoom: 8,
            maxResolution: 1.4
        })
    };

    // Initialize the Map
    OLMap.prototype.initMap = function() {
        var mousePositionControl = this.setMouseControl();
        var baseimage = this.setBaseImageLayer();
        this.map = new ol.Map({
            controls: ol.control.defaults({
            attributionOptions: /** @type {olx.control.AttributionOptions} */ ({
                collapsible: false
            })
            }).extend([mousePositionControl]),
            layers: [baseimage],
            target: this.mapdiv,
            view: this.view
        });
    };

    // Add a Draw Interaction
    OLMap.prototype.addDrawInteraction = function() {
        // set up variable for last saved feature & vector source for point
        var lastFeature;
        var drawsource = new ol.source.Vector({wrapX: false});
        // create new point vectorLayer
        var pointVector = this.newVectorLayer(drawsource);
        // add the layer to the map
        this.map.addLayer(pointVector);

        // New draw event ; default to Point
        var value = 'Point';
        var geometryFunction, maxPoints;
        this.draw = new ol.interaction.Draw({
          source: drawsource,
          type: /** @type {ol.geom.GeometryType} */ (value),
          geometryFunction: geometryFunction,
          maxPoints: maxPoints
        });

        // On draw end, remove the last saved feature (point)
        this.draw.on('drawend', function(e) {
          if (lastFeature) {
            drawsource.removeFeature(lastFeature);
          }
          lastFeature = e.feature;
        });

        // add draw interaction onto the map
        this.map.addInteraction(this.draw);

    };

    // New Vector Layer
    OLMap.prototype.newVectorLayer = function(source) {
        // default set to Point, but eventually expand this to different vector layer types
        var vector = new ol.layer.Vector({
            source: source,
            style: new ol.style.Style({
                fill: new ol.style.Fill({
                    color: 'rgba(255, 255, 255, 0.2)'
                }),
                stroke: new ol.style.Stroke({
                    color: '#ffcc33',
                    width: 2
                }),
                image: new ol.style.Circle({
                    radius: 3,
                    fill: new ol.style.Fill({
                        color: '#ffcc33'
                        })
                    })
            })
        });
        return vector;
    };

    return OLMap;

})();


;/*
* @Author: Brian Cherinka
* @Date:   2016-04-11 14:19:38
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-11 15:24:12
*/

'use strict';

// Javascript code for general things

var Utils,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Utils = (function() {

    marvin.Utils = Utils;

    // Constructor
    function Utils() {

        // in case constructor called without new
        if (false === (this instanceof Utils)) {
            return new Utils();
        }

        this.init();

        // event handlers

    }

    // initialize the object
    Utils.prototype.init = function init() {

    };

    // Build form
    Utils.prototype.buildForm = function buildForm() {
        var _len=arguments.length;
        var args = new Array(_len); for(var $_i = 0; $_i < _len; ++$_i) {args[$_i] = arguments[$_i];}
        var names = args[0];
        var form = {};
        $.each(args.slice(1),function(index,value) {
            form[names[index]] = value;
        });
        return form;
    }

    // Return unique elements of an Array
    Utils.prototype.unique = function unique(data) {
        var result = [];
        $.each(data, function(i, value) {
            if ($.inArray(value, result) == -1) result.push(value);
        });
        return result;
    };

    return Utils;
})();
