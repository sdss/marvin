/*
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


