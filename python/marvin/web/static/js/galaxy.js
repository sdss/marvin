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


