/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-13 17:45:08
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//

'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Galaxy = function () {

    // Constructor

    function Galaxy(plateifu) {
        _classCallCheck(this, Galaxy);

        this.setPlateIfu(plateifu);
        this.maindiv = $('#' + this.plateifu);
        this.mapdiv = this.maindiv.find('#map');
        this.specdiv = this.maindiv.find('#graphdiv');
        this.specmsg = this.maindiv.find('#specmsg');
        this.webspec = null;
        this.staticdiv = this.maindiv.find('#staticdiv');
        this.dynamicdiv = this.maindiv.find('#dynamicdiv');
        this.togglediv = $('#toggleinteract');
    }

    // Test print


    _createClass(Galaxy, [{
        key: 'print',
        value: function print() {
            console.log('We are now printing galaxy', this.plateifu, this.plate, this.ifu);
        }

        // Set the plateifu

    }, {
        key: 'setPlateIfu',
        value: function setPlateIfu(plateifu) {
            if (plateifu === undefined) {
                this.plateifu = $('.galinfo').attr('id');
            } else {
                this.plateifu = plateifu;
            }

            var _plateifu$split = this.plateifu.split('-');

            var _plateifu$split2 = _slicedToArray(_plateifu$split, 2);

            this.plate = _plateifu$split2[0];
            this.ifu = _plateifu$split2[1];
        }

        // Initialize and Load a DyGraph spectrum

    }, {
        key: 'loadSpaxel',
        value: function loadSpaxel(spaxel) {
            this.webspec = new Dygraph(this.specdiv[0], spaxel, {
                labels: ['x', 'Flux'],
                errorBars: true
            });
        }
    }, {
        key: 'updateSpaxel',


        // Update a DyGraph spectrum
        value: function updateSpaxel(spaxel, specmsg) {
            var newmsg = "Here's a spectrum: " + specmsg;
            this.specmsg.empty();
            this.specmsg.html(newmsg);
            this.webspec.updateOptions({ 'file': spaxel });
        }
    }, {
        key: 'initOpenLayers',


        // Initialize OpenLayers Map
        value: function initOpenLayers(image) {
            this.image = image;
            this.olmap = new OLMap(image);
            // add click event handler on map to get spaxel
            this.olmap.map.on('singleclick', this.getSpaxel, this);
        }
    }, {
        key: 'getSpaxel',


        // Retrieves a new Spaxel from the server based on a given mouse position
        value: function getSpaxel(event) {
            var map = event.map;
            var mousecoords = event.coordinate;
            var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords'];
            var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords);
            var _this = this;

            // send the form data
            $.post(Flask.url_for('galaxy_page.getspaxel'), form, 'json').done(function (data) {
                $('#mouse-output').empty();
                var myhtml = "<h5>My mouse coords " + mousecoords + ", message: " + data.result.message + "</h5>";
                $('#mouse-output').html(myhtml);
                _this.updateSpaxel(data.result.spectra, data.result.specmsg);
            }).fail(function (data) {
                $('#mouse-output').empty();
                var myhtml = "<h5>Error message: " + data.result.message + "</h5>";
                $('#mouse-output').html(myhtml);
            });
        }
    }, {
        key: 'toggleInteract',


        // Toggle the interactive OpenLayers map and Dygraph spectra
        value: function toggleInteract(spaxel, image) {
            if (this.togglediv.hasClass('active')) {
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
        }
    }]);

    return Galaxy;
}();
