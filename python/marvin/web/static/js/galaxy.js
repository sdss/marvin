/*
* @Author: Brian Cherinka
* @Date:   2016-04-10 11:42:17
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-11 10:12:24
*/

'use strict';

var Galaxy,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Galaxy = (function () {

    marvin.Galaxy = Galaxy;

    function Galaxy(plateifu) {

        // in case constructor called without new
        if (false === (this instanceof Galaxy)) {
            return new Galaxy();
        }

        this.init(plateifu);

        // Event Handlers
    }

    // initialize the object
    Galaxy.prototype.init = function(plateifu) {
        this.setPlateIfu(plateifu);
        this.maindiv = $('#'+this.plateifu);
        this.mapdiv = this.maindiv.find('#map')[0];
        this.specdiv = this.maindiv.find('#graphdiv')[0];
        this.specmsg = this.maindiv.find('#specmsg')[0];
        this.webspec = null;

        //this.initOpenLayers();
        this.initOpenLayers();

    };

    // test print
    Galaxy.prototype.print = function() {
        console.log('We are now printing galaxy', this.plateifu, this.plate, this.ifu);
    };

    // Determine and Set the plateifu from input
    Galaxy.prototype.setPlateIfu = function(plateifu) {
        if (plateifu === undefined) {
            this.plateifu = $('.galinfo').attr('id');
        } else {
            this.plateifu = plateifu;
        }
        [this.plate, this.ifu] = this.plateifu.split('-');
    };

    // Initialize and Load a DyGraph spectrum
    Galaxy.prototype.loadSpaxel = function(spaxel) {
        this.webspec = new Dygraph(this.specdiv,
                  spaxel,
                  {
                    labels: ['x','Flux'],
                    errorBars: true
                  });
    };

    // Update a DyGraph spectrum
    Galaxy.prototype.updateSpaxel = function(spaxel, specmsg) {
        var newmsg = "Here's a spectrum: "+specmsg;
        this.specmsg.html(newmsg);
        this.webspec.updateOptions({'file': spaxel});
    };

    // Initialize OpenLayers Map
    Galaxy.prototype.initOpenLayers = function() {
      var mousePositionControl = new ol.control.MousePosition({
        coordinateFormat: ol.coordinate.createStringXY(4),
        projection: 'EPSG:4326',
        // comment the following two lines to have the mouse position
        // be placed within the map.
        className: 'custom-mouse-position',
        target: document.getElementById('mouse-position'),
        undefinedHTML: '&nbsp;'
      });

      var extent = [0, 0, 562, 562];
      var projection = new ol.proj.Projection({
        code: 'ifu',
        units: 'pixels',
        extent: extent
      });

      var lastFeature;
      var source = new ol.source.Vector({wrapX: false});

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

      var map = new ol.Map({
        controls: ol.control.defaults({
          attributionOptions: /** @type {olx.control.AttributionOptions} */ ({
            collapsible: false
          })
        }).extend([mousePositionControl]),
        layers: [
          new ol.layer.Image({
            source: new ol.source.ImageStatic({
              url: 'http://localhost:80/sas/mangawork/manga/spectro/redux/v1_5_1/8485/stack/images/test_wcs2.png',
              projection: projection,
              imageExtent: extent
            })
          })
        , vector],
        target: 'map',
        view: new ol.View({
          projection: projection,
          center: ol.extent.getCenter(extent),
          zoom: 0,
          maxZoom: 8,
          maxResolution: 1.4
        })
      });

      var draw;
      function addInteraction() {
        var value = 'Point';
        var geometryFunction, maxPoints;
        draw = new ol.interaction.Draw({
          source: source,
          type: /** @type {ol.geom.GeometryType} */ (value),
          geometryFunction: geometryFunction,
          maxPoints: maxPoints
        });

        draw.on('drawend', function(e) {
          if (lastFeature) {
            source.removeFeature(lastFeature);
          }
          lastFeature = e.feature;
        });

        map.addInteraction(draw);
      };

      addInteraction();

    };

    return Galaxy;

})();


