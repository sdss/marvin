/*
* @Author: Brian Cherinka
* @Date:   2016-04-29 09:29:24
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-29 09:45:04
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Carousel = function () {

    // Constructor
    function Carousel(cardiv, thumbs) {
        _classCallCheck(this, Carousel);

        this.carouseldiv = $(cardiv);
        this.thumbsdiv = thumbs !== undefined ? $(thumbs) : $('[id^=carousel-selector-]');

        // init the carousel
        this.carouseldiv.carousel({
            interval: 5000
        });

        // Event handlers
        this.thumbsdiv.on('click', this, this.handleThumbs);
        this.carouseldiv.on('slid.bs.carousel', this, this.updateText);
    }

    // Print


    _createClass(Carousel, [{
        key: 'print',
        value: function print() {
            console.log('I am Carousel!');
        }

        // Handle the carousel thumbnails

    }, {
        key: 'handleThumbs',
        value: function handleThumbs(event) {
            var _this = event.data;
            var id_selector = $(this).attr("id");
            try {
                var id = /-(\d+)$/.exec(id_selector)[1];
                //console.log(id_selector, id);
                _this.carouseldiv.carousel(parseInt(id));
            } catch (e) {
                console.log('MyCarousel: Regex failed!', e);
            }
        }

        // When carousel slides, auto update the text

    }, {
        key: 'updateText',
        value: function updateText(event) {
            var _this = event.data;
            var id = $('.item.active').data('slide-number');
            $('#carousel-text').html($('#slide-content-' + id).html());
        }
    }]);

    return Carousel;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian Cherinka
<<<<<<< HEAD
* @Last Modified time: 2016-11-05 14:53:52
=======
* @Last Modified time: 2016-09-26 17:40:15
>>>>>>> upstream/marvin_refactor
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
    function Galaxy(plateifu, toggleon) {
        _classCallCheck(this, Galaxy);

        this.setPlateIfu(plateifu);
        this.toggleon = toggleon;
        this.maindiv = $('#' + this.plateifu);
        this.metadiv = this.maindiv.find('#metadata');
        this.specdiv = this.maindiv.find('#specview');
        this.imagediv = this.specdiv.find('#imagediv');
        this.mapsdiv = this.specdiv.find('#mapsdiv');
        this.mapdiv = this.specdiv.find('#mapdiv1');
        this.graphdiv = this.specdiv.find('#graphdiv');
        this.specmsg = this.specdiv.find('#specmsg');
        this.mapmsg = this.specdiv.find('#mapmsg');
        this.webspec = null;
        this.staticdiv = this.specdiv.find('#staticdiv');
        this.dynamicdiv = this.specdiv.find('#dynamicdiv');
        this.togglediv = $('#toggleinteract');
        this.toggleload = $('#toggle-load');
        this.togglediv.bootstrapToggle('off');
        this.qualpop = $('#qualitypopover');
        this.targpops = $('.targpopovers');
        this.dapmapsbut = $('#dapmapsbut');
        this.dapselect = $('#dapmapchoices');
        this.dapbt = $('#dapbtchoices');
        this.dapselect.selectpicker('deselectAll');
        this.resetmapsbut = $('#resetmapsbut');

        // init some stuff
        this.initFlagPopovers();
        //this.checkToggle();

        //Event Handlers
        this.dapmapsbut.on('click', this, this.getDapMaps);
        this.resetmapsbut.on('click', this, this.resetMaps);
        this.togglediv.on('change', this, this.initDynamic);
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
                this.plateifu = $('.singlegalaxy').attr('id');
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
        value: function loadSpaxel(spaxel, title) {
            var labels = spaxel[0].length == 3 ? ['Wavelength', 'Flux', 'Model Fit'] : ['Wavelength', 'Flux'];
            this.webspec = new Dygraph(this.graphdiv[0], spaxel, {
                title: title,
                labels: labels,
                errorBars: true, // TODO DyGraph shows 2-sigma error bars FIX THIS
                ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
                xlabel: 'Wavelength [Ångströms]'
            });
        }
    }, {
        key: 'updateSpecMsg',


        // Update the spectrum message div for errors only
        value: function updateSpecMsg(specmsg, status) {
            this.specmsg.hide();
            if (status !== undefined && status === -1) {
                this.specmsg.show();
            }
            var newmsg = '<strong>' + specmsg + '</strong>';
            this.specmsg.empty();
            this.specmsg.html(newmsg);
        }

        // Update a DyGraph spectrum

    }, {
        key: 'updateSpaxel',
        value: function updateSpaxel(spaxel, specmsg) {
            this.updateSpecMsg(specmsg);
            this.webspec.updateOptions({ 'file': spaxel, 'title': specmsg });
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
        key: 'initHeatmap',
        value: function initHeatmap(maps) {
            console.log('initHeatmap', this.mapsdiv);
            var mapchildren = this.mapsdiv.children('div');
            console.log('mapchildren', mapchildren);
            var _this = this;
            $.each(mapchildren, function (index, child) {
                var mapdiv = $(child).find('div').first();
                mapdiv.empty();
                if (maps[index] !== undefined && maps[index].data !== null) {
                    this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg, _this);
                    this.heatmap.mapdiv.highcharts().reflow();
                }
            });
        }
    }, {
        key: 'getSpaxel',


        // Retrieves a new Spaxel from the server based on a given mouse position or xy spaxel coord.
        value: function getSpaxel(event) {
            var mousecoords = event.coordinate === undefined ? null : event.coordinate;
            var divid = $(event.target).parents('div').first().attr('id');
            var maptype = divid !== undefined && divid.search('highcharts') !== -1 ? 'heatmap' : 'optical';
            var x = event.point === undefined ? null : event.point.x;
            var y = event.point === undefined ? null : event.point.y;
            var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords', 'type', 'x', 'y'];
            var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords, maptype, x, y);
            var _this = this;

            // send the form data
            $.post(Flask.url_for('galaxy_page.getspaxel'), form, 'json').done(function (data) {
                if (data.result.status !== -1) {
                    _this.updateSpaxel(data.result.spectra, data.result.specmsg);
                } else {
                    _this.updateSpecMsg('Error: ' + data.result.specmsg, data.result.status);
                }
            }).fail(function (data) {
                _this.updateSpecMsg('Error: ' + data.result.specmsg, data.result.status);
            });
        }
    }, {
        key: 'checkToggle',


        // check the toggle preference on initial page load
        // eventually for user preferences
        value: function checkToggle() {
            if (this.toggleon) {
                this.toggleOn();
            } else {
                this.toggleOff();
            }
        }

        // toggle the display button on

    }, {
        key: 'toggleOn',
        value: function toggleOn() {
            // eventually this should include the ajax stuff inside initDynamic - for after user preferences implemented
            this.toggleon = true;
            //this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
            //this.togglediv.button('complete');
            this.staticdiv.hide();
            this.dynamicdiv.show();
        }

        // toggle the display button off

    }, {
        key: 'toggleOff',
        value: function toggleOff() {
            this.toggleon = false;
            //this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
            //this.togglediv.button('reset');
            this.dynamicdiv.hide();
            this.staticdiv.show();
        }
    }, {
        key: 'testTogg',
        value: function testTogg(event) {
            var _this = event.data;
            console.log('toggling', _this.togglediv.prop('checked'), _this.togglediv.hasClass('active'));
        }

        // Initialize the Dynamic Galaxy Interaction upon toggle - makes loading an AJAX request

    }, {
        key: 'initDynamic',
        value: function initDynamic(event) {

            var _this = event.data;

            if (!_this.togglediv.prop('checked')) {
                // Turning Off
                _this.toggleOff();
            } else {
                // Turning On
                _this.toggleOn();

                // check for empty divs
                var specempty = _this.graphdiv.is(':empty');
                var imageempty = _this.imagediv.is(':empty');
                var mapempty = _this.mapdiv.is(':empty');

                // send the request if the dynamic divs are empty
                if (imageempty) {
                    // make the form
                    var keys = ['plateifu', 'toggleon'];
                    var form = m.utils.buildForm(keys, _this.plateifu, _this.toggleon);
                    _this.toggleload.show();

                    $.post(Flask.url_for('galaxy_page.initdynamic'), form, 'json').done(function (data) {

                        var image = data.result.image;
                        var spaxel = data.result.spectra;
                        var spectitle = data.result.specmsg;
                        var maps = data.result.maps;
                        var mapmsg = data.result.mapmsg;

                        // Load the Galaxy Image
                        _this.initOpenLayers(image);
                        _this.toggleload.hide();

                        // Try to load the spaxel
                        if (data.result.specstatus !== -1) {
                            _this.loadSpaxel(spaxel, spectitle);
                        } else {
                            _this.updateSpecMsg('Error: ' + spectitle, data.result.specstatus);
                        }

                        // Try to load the Maps
                        if (data.result.mapstatus !== -1) {
                            _this.initHeatmap(maps);
                        } else {
                            _this.updateMapMsg('Error: ' + mapmsg, data.result.mapstatus);
                        }
                    }).fail(function (data) {
                        _this.updateSpecMsg('Error: ' + data.result.specmsg, data.result.specstatus);
                        _this.updateMapMsg('Error: ' + data.result.mapmsg, data.result.mapstatus);
                        _this.toggleload.hide();
                    });
                }
            }
        }

        // Toggle the interactive OpenLayers map and Dygraph spectra
        // DEPRECATED - REMOVE

    }, {
        key: 'toggleInteract',
        value: function toggleInteract(image, maps, spaxel, spectitle, mapmsg) {
            if (this.togglediv.hasClass('active')) {
                // Turning Off
                this.toggleon = false;
                this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
                this.togglediv.button('reset');
                this.dynamicdiv.hide();
                this.staticdiv.show();
            } else {
                // Turning On
                this.toggleon = true;
                this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
                this.togglediv.button('complete');
                this.staticdiv.hide();
                this.dynamicdiv.show();

                // check for empty divs
                var specempty = this.graphdiv.is(':empty');
                var imageempty = this.imagediv.is(':empty');
                var mapempty = this.mapdiv.is(':empty');
                // load the spaxel if the div is initially empty;
                if (this.graphdiv !== undefined && specempty) {
                    this.loadSpaxel(spaxel, spectitle);
                }
                // load the image if div is empty
                if (imageempty) {
                    this.initOpenLayers(image);
                }
                // load the map if div is empty
                if (mapempty) {
                    this.initHeatmap(maps);
                }

                // possibly update an initial map message
                if (mapmsg !== null) {
                    this.updateMapMsg(mapmsg, -1);
                }
            }
        }
    }, {
        key: 'initFlagPopovers',


        //  Initialize the Quality and Target Popovers
        value: function initFlagPopovers() {
            // DRP Quality Popovers
            this.qualpop.popover({ html: true, content: $('#list_drp3quality').html() });
            // MaNGA Target Popovers
            $.each(this.targpops, function (index, value) {
                // get id of flag link
                var popid = value.id;
                // split id and grab the mngtarg

                var _popid$split = popid.split('_');

                var _popid$split2 = _slicedToArray(_popid$split, 2);

                var base = _popid$split2[0];
                var targ = _popid$split2[1];
                // build the label list id

                var listid = '#list_' + targ;
                // init the specific popover
                $('#' + popid).popover({ html: true, content: $(listid).html() });
            });
        }
    }, {
        key: 'getDapMaps',


        // Get some DAP Maps
        value: function getDapMaps(event) {
            var _this = event.data;
            console.log('getting dap maps', _this.dapselect.selectpicker('val'));
            var params = _this.dapselect.selectpicker('val');
            var bintemp = _this.dapbt.selectpicker('val');
            var keys = ['plateifu', 'params', 'bintemp'];
            var form = m.utils.buildForm(keys, _this.plateifu, params, bintemp);
            _this.mapmsg.hide();
            $(this).button('loading');

            // send the form data
            $.post(Flask.url_for('galaxy_page.updatemaps'), form, 'json').done(function (data) {
                if (data.result.status !== -1) {
                    _this.dapmapsbut.button('reset');
                    _this.initHeatmap(data.result.maps);
                } else {
                    _this.updateMapMsg('Error: ' + data.result.mapmsg, data.result.status);
                    _this.dapmapsbut.button('reset');
                }
            }).fail(function (data) {
                _this.updateMapMsg('Error: ' + data.result.mapmsg, data.result.status);
                _this.dapmapsbut.button('reset');
            });
        }
    }, {
        key: 'updateMapMsg',


        // Update the Map Msg
        value: function updateMapMsg(mapmsg, status) {
            this.mapmsg.hide();
            if (status !== undefined && status === -1) {
                this.mapmsg.show();
            }
            var newmsg = '<strong>' + mapmsg + '</strong>';
            this.mapmsg.empty();
            this.mapmsg.html(newmsg);
        }
    }, {
        key: 'resetMaps',


        // Reset the Maps selection
        value: function resetMaps(event) {
            var _this = event.data;
            _this.mapmsg.hide();
            _this.dapselect.selectpicker('deselectAll');
            _this.dapselect.selectpicker('refresh');
        }
    }]);

    return Galaxy;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-10-12 16:44:53
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Header = function () {

    // Constructor
    function Header() {
        _classCallCheck(this, Header);

        this.navbar = $('.navbar');
        this.galidform = $('#headform');
        this.typeahead = $('#headform .typeahead');
        this.mplform = $('#mplform');
        this.mplselect = $('#mplselect');

        this.initTypeahead();

        //Event Handlers
        this.mplselect.on('change', this, this.selectMPL);
    }

    // Print


    _createClass(Header, [{
        key: 'print',
        value: function print() {
            console.log('I am Header!');
        }

        // Initialize the Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead(typediv, formdiv, url, fxn) {

            var _this = this;
            var typediv = typediv === undefined ? this.typeahead : $(typediv);
            var formdiv = formdiv === undefined ? this.galidform : $(formdiv);
            var typeurl = url === undefined ? Flask.url_for('index_page.getgalidlist') : url;
            var afterfxn = fxn === undefined ? null : fxn;

            // create the bloodhound engine
            this.galids = new Bloodhound({
                datumTokenizer: Bloodhound.tokenizers.whitespace,
                queryTokenizer: Bloodhound.tokenizers.whitespace,
                //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
                prefetch: typeurl,
                remote: {
                    url: typeurl,
                    filter: function filter(galids) {
                        return galids;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.galids.initialize();

            typediv.typeahead('destroy');
            typediv.typeahead({
                showHintOnFocus: true,
                items: 30,
                source: this.galids.ttAdapter(),
                afterSelect: function afterSelect() {
                    formdiv.submit();
                }
            });
        }

        // Select the MPL version on the web

    }, {
        key: 'selectMPL',
        value: function selectMPL(event) {
            var _this = event.data;
            var url = 'index_page.selectmpl';
            var verform = m.utils.serializeForm('#mplform');
            console.log('setting new mpl', verform);
            _this.sendAjax(verform, url, _this.reloadPage);
        }

        // Reload the Current Page

    }, {
        key: 'reloadPage',
        value: function reloadPage() {
            location.reload(true);
        }

        // Send an AJAX request

    }, {
        key: 'sendAjax',
        value: function sendAjax(form, url, fxn) {
            var _this = this;
            $.post(Flask.url_for(url), form, 'json').done(function (data) {
                // reload the current page, this re-instantiates a new Header with new version info from session
                if (data.result.status == 1) {
                    fxn();
                    _this.galids.clearPrefetchCache();
                    _this.galids.initialize();
                } else {
                    alert('Failed to set the versions! ' + data.result.msg);
                }
            }).fail(function (data) {
                alert('Failed to set the versions! Problem with Flask setversion. ' + data.result.msg);
            });
        }
    }]);

    return Header;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-11-05 15:07:00
*/

'use strict';

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var HeatMap = function () {

    // Constructor
    function HeatMap(mapdiv, data, title, galthis) {
        _classCallCheck(this, HeatMap);

        if (data === undefined) {
            console.error('Must specify input map data to initialize a HeatMap!');
        } else if (mapdiv === undefined) {
            console.error('Must specify an input mapdiv to initialize a HeatMap');
        } else {
            this.mapdiv = mapdiv; // div element for map
            this.data = data; // map data
            this.title = title; // map title
            this.galthis = galthis; //the self of the Galaxy class
            this.parseTitle();
            this.initMap();
            this.setColorNoData(this, Highcharts);
        }
    }

    _createClass(HeatMap, [{
        key: 'print',


        // test print
        value: function print() {
            console.log('We are now printing heatmap for ', this.title);
        }
    }, {
        key: 'parseTitle',


        // Parse the heatmap title into category, parameter, channel
        // e.g. 7443-1901: emline_gflux_ha-6564
        value: function parseTitle() {
            var _title$split = this.title.split(':');

            var _title$split2 = _slicedToArray(_title$split, 2);

            var plateifu = _title$split2[0];
            var newtitle = _title$split2[1];

            var _newtitle$split = newtitle.split('_');

            var _newtitle$split2 = _slicedToArray(_newtitle$split, 3);

            this.category = _newtitle$split2[0];
            this.parameter = _newtitle$split2[1];
            this.channel = _newtitle$split2[2];
        }

        // Get range of x (or y) data and z (DAP property) data

    }, {
        key: 'getRange',
        value: function getRange() {
            var xylength = this.data['values'].length;
            var xyrange = Array.apply(null, { length: xylength }).map(Number.call, Number);
            var zrange = [].concat.apply([], this.data['values']);
            return [xyrange, zrange];
        }

        // Filter out null and no-data from z (DAP prop) data

    }, {
        key: 'filterRange',
        value: function filterRange(z) {
            if (z !== undefined && typeof z === 'number' && !isNaN(z)) {
                return true;
            } else {
                return false;
            }
        }

        // return the min and max of a range

    }, {
        key: 'getMinMax',
        value: function getMinMax(range) {
            // var range = (range === undefined) ? this.getRange() : range;
            var min = Math.min.apply(null, range);
            var max = Math.max.apply(null, range);
            return [min, max];
        }
    }, {
        key: 'setNull',
        value: function setNull(x) {
            var values = x.values;
            var ivar = x.ivar;
            var mask = x.mask;

            var xyz = Array();

            for (var ii = 0; ii < values.length; ii++) {
                for (var jj = 0; jj < values.length; jj++) {
                    var val = values[ii][jj];

                    if (mask !== null) {
                        var noValue = mask[ii][jj] && Math.pow(2, 0);
                        var badValue = mask[ii][jj] && Math.pow(2, 5);
                        var mathError = mask[ii][jj] && Math.pow(2, 6);
                        var badFit = mask[ii][jj] && Math.pow(2, 7);
                        var doNotUse = mask[ii][jj] && Math.pow(2, 30);
                        //var noData = (noValue || badValue || mathError || badFit || doNotUse);
                        var noData = noValue;
                        var badData = badValue || mathError || badFit || doNotUse;
                    } else {
                        noData == null;
                        badData == null;
                    }

                    if (ivar !== null) {
                        var signalToNoise = Math.abs(val) * Math.sqrt(ivar[ii][jj]);
                        var signalToNoiseThreshold = 1.;
                    }

                    // value types
                    // val=no-data => gray color
                    // val=null => hatch area
                    // val=low-sn => nothing at the moment

                    if (noData) {
                        // for data that is outside the range "nocov" mask
                        val = 'no-data';
                    } else if (badData) {
                        // for data that is bad - masked in some way
                        val = null;
                    } else if (ivar !== null && signalToNoise < signalToNoiseThreshold) {
                        // for data that is low S/N
                        val = null; //val = 'low-sn';
                    } else if (ivar === null) {
                        // for data with no mask or no inverse variance extensions
                        if (this.title.search('binid') !== -1) {
                            // for binid extension only, set -1 values to no data
                            val = val == -1 ? 'no-data' : val;
                        } else if (val === 0.0) {
                            // set zero values to no-data
                            val = 'no-data';
                        }
                    };
                    xyz.push([ii, jj, val]);
                };
            };
            return xyz;
        }
    }, {
        key: 'setColorNoData',
        value: function setColorNoData(_this, H) {
            H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
                if (value === 'no-data') {
                    // make gray color
                    return 'rgba(0,0,0,0)'; // '#A8A8A8';
                } else if (value === 'low-sn') {
                    // make light blue with half-opacity == muddy blue-gray
                    return 'rgba(0,191,255,0.5)'; //'#7fffd4';
                } else return proceed.apply(this, Array.prototype.slice.call(arguments, 1));
            });
        }
    }, {
        key: 'setColorMapHex',
        value: function setColorMapHex(cmap) {

            var linearLabHex = ['#040404', '#0a0308', '#0d040b', '#10050e', '#120510', '#150612', '#160713', '#180815', '#1a0816', '#1b0918', '#1c0a19', '#1e0b1a', '#1f0c1b', '#200c1c', '#210d1d', '#230e1f', '#240e20', '#250f20', '#260f21', '#271022', '#281123', '#291124', '#2a1226', '#2b1326', '#2c1327', '#2e1429', '#2e142d', '#2e1532', '#2d1537', '#2d153c', '#2d1640', '#2d1743', '#2d1747', '#2d184b', '#2d184d', '#2d1951', '#2d1954', '#2c1a57', '#2c1b5a', '#2d1b5c', '#2d1c5f', '#2c1d62', '#2c1d64', '#2c1e67', '#2c1f6a', '#2c1f6d', '#2c206e', '#2c2171', '#2c2274', '#2c2276', '#2a2379', '#282678', '#262877', '#242a78', '#222c78', '#212e78', '#202f78', '#1f3179', '#1e327a', '#1e337b', '#1d347b', '#1d357d', '#1c377d', '#1c387e', '#1b397f', '#1c3a80', '#1c3b81', '#1b3c81', '#1b3d83', '#1b3e84', '#1b3f85', '#1c4086', '#1b4187', '#1b4288', '#1b4489', '#1b458a', '#194788', '#164986', '#154a85', '#144c83', '#114e81', '#104f80', '#0f517e', '#0e527d', '#0a547b', '#0a557a', '#095778', '#085877', '#075976', '#065b75', '#045c73', '#045e72', '#045f72', '#036070', '#01626f', '#01636e', '#00646d', '#00656c', '#00676b', '#00686a', '#006969', '#006b68', '#006c65', '#006e64', '#006f63', '#007062', '#007260', '#00735f', '#00745d', '#00765c', '#00775a', '#007859', '#007958', '#007b56', '#007c55', '#007d53', '#007f52', '#008050', '#00814f', '#00834d', '#00844b', '#008549', '#008648', '#008846', '#008944', '#008a42', '#008b41', '#008d40', '#008e3f', '#008f3d', '#00913c', '#00923c', '#00933a', '#009539', '#009638', '#009737', '#009935', '#009a34', '#009b33', '#009d32', '#009e30', '#009f2f', '#00a02d', '#00a22c', '#00a32a', '#00a429', '#00a527', '#00a724', '#00a822', '#00a91f', '#00aa17', '#00a908', '#09aa00', '#14ab00', '#1dac00', '#23ad00', '#28ae00', '#2daf00', '#30b000', '#34b100', '#37b200', '#3bb300', '#3db400', '#40b500', '#42b600', '#44b700', '#47b800', '#49b900', '#4cba00', '#4ebb00', '#4fbc00', '#51bd00', '#53be00', '#55bf00', '#57c000', '#5cc000', '#63c100', '#6ac100', '#72c100', '#77c200', '#7dc200', '#82c200', '#87c300', '#8cc300', '#91c300', '#95c400', '#99c400', '#9dc500', '#a1c500', '#a5c500', '#a9c600', '#acc600', '#b0c700', '#b4c700', '#b8c700', '#bac800', '#bec900', '#c1c900', '#c5c900', '#c8ca00', '#c9c918', '#cbca33', '#ceca41', '#cfcb4d', '#d1cb57', '#d4cb5f', '#d5cc67', '#d7cd6d', '#dacd74', '#dbce79', '#ddcf7f', '#dfcf84', '#e2cf8a', '#e3d08f', '#e5d193', '#e7d197', '#e8d29b', '#ebd39f', '#edd3a4', '#eed4a8', '#f0d4ac', '#f3d5af', '#f3d6b3', '#f5d6b7', '#f8d7ba', '#f8d8bd', '#f8dac1', '#f7dbc3', '#f7dcc6', '#f7dec9', '#f8dfcc', '#f7e0ce', '#f7e2d1', '#f7e3d3', '#f7e5d6', '#f7e6d8', '#f7e7da', '#f7e8dc', '#f8eae0', '#f7ebe1', '#f7ece5', '#f7eee7', '#f7efe8', '#f8f0eb', '#f8f2ed', '#f7f3ef', '#f8f4f1', '#f8f6f4', '#f8f7f6', '#f8f8f8', '#f9f9f9', '#fbfbfb', '#fcfcfc', '#fdfdfd', '#fefefe', '#ffffff'];

            var infernoHex = ['#000004', '#010005', '#010106', '#010108', '#02010a', '#02020c', '#02020e', '#030210', '#040312', '#040314', '#050417', '#060419', '#07051b', '#08051d', '#09061f', '#0a0722', '#0b0724', '#0c0826', '#0d0829', '#0e092b', '#10092d', '#110a30', '#120a32', '#140b34', '#150b37', '#160b39', '#180c3c', '#190c3e', '#1b0c41', '#1c0c43', '#1e0c45', '#1f0c48', '#210c4a', '#230c4c', '#240c4f', '#260c51', '#280b53', '#290b55', '#2b0b57', '#2d0b59', '#2f0a5b', '#310a5c', '#320a5e', '#340a5f', '#360961', '#380962', '#390963', '#3b0964', '#3d0965', '#3e0966', '#400a67', '#420a68', '#440a68', '#450a69', '#470b6a', '#490b6a', '#4a0c6b', '#4c0c6b', '#4d0d6c', '#4f0d6c', '#510e6c', '#520e6d', '#540f6d', '#550f6d', '#57106e', '#59106e', '#5a116e', '#5c126e', '#5d126e', '#5f136e', '#61136e', '#62146e', '#64156e', '#65156e', '#67166e', '#69166e', '#6a176e', '#6c186e', '#6d186e', '#6f196e', '#71196e', '#721a6e', '#741a6e', '#751b6e', '#771c6d', '#781c6d', '#7a1d6d', '#7c1d6d', '#7d1e6d', '#7f1e6c', '#801f6c', '#82206c', '#84206b', '#85216b', '#87216b', '#88226a', '#8a226a', '#8c2369', '#8d2369', '#8f2469', '#902568', '#922568', '#932667', '#952667', '#972766', '#982766', '#9a2865', '#9b2964', '#9d2964', '#9f2a63', '#a02a63', '#a22b62', '#a32c61', '#a52c60', '#a62d60', '#a82e5f', '#a92e5e', '#ab2f5e', '#ad305d', '#ae305c', '#b0315b', '#b1325a', '#b3325a', '#b43359', '#b63458', '#b73557', '#b93556', '#ba3655', '#bc3754', '#bd3853', '#bf3952', '#c03a51', '#c13a50', '#c33b4f', '#c43c4e', '#c63d4d', '#c73e4c', '#c83f4b', '#ca404a', '#cb4149', '#cc4248', '#ce4347', '#cf4446', '#d04545', '#d24644', '#d34743', '#d44842', '#d54a41', '#d74b3f', '#d84c3e', '#d94d3d', '#da4e3c', '#db503b', '#dd513a', '#de5238', '#df5337', '#e05536', '#e15635', '#e25734', '#e35933', '#e45a31', '#e55c30', '#e65d2f', '#e75e2e', '#e8602d', '#e9612b', '#ea632a', '#eb6429', '#eb6628', '#ec6726', '#ed6925', '#ee6a24', '#ef6c23', '#ef6e21', '#f06f20', '#f1711f', '#f1731d', '#f2741c', '#f3761b', '#f37819', '#f47918', '#f57b17', '#f57d15', '#f67e14', '#f68013', '#f78212', '#f78410', '#f8850f', '#f8870e', '#f8890c', '#f98b0b', '#f98c0a', '#f98e09', '#fa9008', '#fa9207', '#fa9407', '#fb9606', '#fb9706', '#fb9906', '#fb9b06', '#fb9d07', '#fc9f07', '#fca108', '#fca309', '#fca50a', '#fca60c', '#fca80d', '#fcaa0f', '#fcac11', '#fcae12', '#fcb014', '#fcb216', '#fcb418', '#fbb61a', '#fbb81d', '#fbba1f', '#fbbc21', '#fbbe23', '#fac026', '#fac228', '#fac42a', '#fac62d', '#f9c72f', '#f9c932', '#f9cb35', '#f8cd37', '#f8cf3a', '#f7d13d', '#f7d340', '#f6d543', '#f6d746', '#f5d949', '#f5db4c', '#f4dd4f', '#f4df53', '#f4e156', '#f3e35a', '#f3e55d', '#f2e661', '#f2e865', '#f2ea69', '#f1ec6d', '#f1ed71', '#f1ef75', '#f1f179', '#f2f27d', '#f2f482', '#f3f586', '#f3f68a', '#f4f88e', '#f5f992', '#f6fa96', '#f8fb9a', '#f9fc9d', '#fafda1', '#fcffa4'];

            var RdBuHex = ['#053061', '#063264', '#073467', '#08366a', '#09386d', '#0a3b70', '#0c3d73', '#0d3f76', '#0e4179', '#0f437b', '#10457e', '#114781', '#124984', '#134c87', '#144e8a', '#15508d', '#175290', '#185493', '#195696', '#1a5899', '#1b5a9c', '#1c5c9f', '#1d5fa2', '#1e61a5', '#1f63a8', '#2065ab', '#2267ac', '#2369ad', '#246aae', '#266caf', '#276eb0', '#2870b1', '#2a71b2', '#2b73b3', '#2c75b4', '#2e77b5', '#2f79b5', '#307ab6', '#327cb7', '#337eb8', '#3480b9', '#3681ba', '#3783bb', '#3885bc', '#3a87bd', '#3b88be', '#3c8abe', '#3e8cbf', '#3f8ec0', '#408fc1', '#4291c2', '#4393c3', '#4695c4', '#4997c5', '#4c99c6', '#4f9bc7', '#529dc8', '#569fc9', '#59a1ca', '#5ca3cb', '#5fa5cd', '#62a7ce', '#65a9cf', '#68abd0', '#6bacd1', '#6eaed2', '#71b0d3', '#75b2d4', '#78b4d5', '#7bb6d6', '#7eb8d7', '#81bad8', '#84bcd9', '#87beda', '#8ac0db', '#8dc2dc', '#90c4dd', '#93c6de', '#96c7df', '#98c8e0', '#9bc9e0', '#9dcbe1', '#a0cce2', '#a2cde3', '#a5cee3', '#a7d0e4', '#a9d1e5', '#acd2e5', '#aed3e6', '#b1d5e7', '#b3d6e8', '#b6d7e8', '#b8d8e9', '#bbdaea', '#bddbea', '#c0dceb', '#c2ddec', '#c5dfec', '#c7e0ed', '#cae1ee', '#cce2ef', '#cfe4ef', '#d1e5f0', '#d2e6f0', '#d4e6f1', '#d5e7f1', '#d7e8f1', '#d8e9f1', '#dae9f2', '#dbeaf2', '#ddebf2', '#deebf2', '#e0ecf3', '#e1edf3', '#e3edf3', '#e4eef4', '#e6eff4', '#e7f0f4', '#e9f0f4', '#eaf1f5', '#ecf2f5', '#edf2f5', '#eff3f5', '#f0f4f6', '#f2f5f6', '#f3f5f6', '#f5f6f7', '#f6f7f7', '#f7f6f6', '#f7f5f4', '#f8f4f2', '#f8f3f0', '#f8f2ef', '#f8f1ed', '#f9f0eb', '#f9efe9', '#f9eee7', '#f9ede5', '#f9ebe3', '#faeae1', '#fae9df', '#fae8de', '#fae7dc', '#fbe6da', '#fbe5d8', '#fbe4d6', '#fbe3d4', '#fce2d2', '#fce0d0', '#fcdfcf', '#fcdecd', '#fdddcb', '#fddcc9', '#fddbc7', '#fdd9c4', '#fcd7c2', '#fcd5bf', '#fcd3bc', '#fbd0b9', '#fbceb7', '#fbccb4', '#facab1', '#fac8af', '#f9c6ac', '#f9c4a9', '#f9c2a7', '#f8bfa4', '#f8bda1', '#f8bb9e', '#f7b99c', '#f7b799', '#f7b596', '#f6b394', '#f6b191', '#f6af8e', '#f5ac8b', '#f5aa89', '#f5a886', '#f4a683', '#f3a481', '#f2a17f', '#f19e7d', '#f09c7b', '#ef9979', '#ee9677', '#ec9374', '#eb9172', '#ea8e70', '#e98b6e', '#e8896c', '#e6866a', '#e58368', '#e48066', '#e37e64', '#e27b62', '#e17860', '#df765e', '#de735c', '#dd7059', '#dc6e57', '#db6b55', '#da6853', '#d86551', '#d7634f', '#d6604d', '#d55d4c', '#d35a4a', '#d25849', '#d05548', '#cf5246', '#ce4f45', '#cc4c44', '#cb4942', '#c94741', '#c84440', '#c6413e', '#c53e3d', '#c43b3c', '#c2383a', '#c13639', '#bf3338', '#be3036', '#bd2d35', '#bb2a34', '#ba2832', '#b82531', '#b72230', '#b61f2e', '#b41c2d', '#b3192c', '#b1182b', '#ae172a', '#ab162a', '#a81529', '#a51429', '#a21328', '#9f1228', '#9c1127', '#991027', '#960f27', '#930e26', '#900d26', '#8d0c25', '#8a0b25', '#870a24', '#840924', '#810823', '#7f0823', '#7c0722', '#790622', '#760521', '#730421', '#700320', '#6d0220', '#6a011f', '#67001f'];

            if (cmap === "linearLab") {
                return linearLabHex;
            } else if (cmap === "inferno") {
                return infernoHex;
            } else if (cmap === "RdBu") {
                return RdBuHex;
            } else {
                return ["#000000", "#FFFFFF"];
            };
        }
    }, {
        key: 'setColorStops',
        value: function setColorStops(cmap) {
            var colorHex = this.setColorMapHex(cmap);
            var stopLocations = colorHex.length;
            var colormap = new Array(stopLocations);
            for (var ii = 0; ii < stopLocations; ii++) {
                colormap[ii] = [ii / (stopLocations - 1), colorHex[ii]];
            };
            return colormap;
        }
    }, {
        key: 'quantileClip',
        value: function quantileClip(range) {
            var quantLow, quantHigh, zQuantLow, zQuantHigh;

            var _getMinMax = this.getMinMax(range);

            var _getMinMax2 = _slicedToArray(_getMinMax, 2);

            zQuantLow = _getMinMax2[0];
            zQuantHigh = _getMinMax2[1];

            if (this.title.toLowerCase().indexOf("vel") >= 0 || this.title.toLowerCase().indexOf("sigma") >= 0) {
                quantLow = 10;
                quantHigh = 90;
            } else if (this.title.toLowerCase().indexOf("flux") >= 0) {
                quantLow = 5;
                quantHigh = 95;
            };

            if (range.length > 0) {
                if (quantLow > 0) {
                    zQuantLow = math.quantileSeq(range, quantLow / 100);
                }
                if (quantHigh < 100) {
                    zQuantHigh = math.quantileSeq(range, quantHigh / 100);
                }
            }
            return [zQuantLow, zQuantHigh];
        }

        // initialize the heat map

    }, {
        key: 'initMap',
        value: function initMap() {
            // set the galaxy class self to a variable
            var _galthis = this.galthis;

            // get the ranges
            //var range  = this.getXRange();
            var xyrange, zrange;

            // get the min and max of the ranges
            var _getRange = this.getRange();

            var _getRange2 = _slicedToArray(_getRange, 2);

            xyrange = _getRange2[0];
            zrange = _getRange2[1];
            var xymin, xymax, zmin, zmax;

            var _getMinMax3 = this.getMinMax(xyrange);

            var _getMinMax4 = _slicedToArray(_getMinMax3, 2);

            xymin = _getMinMax4[0];
            xymax = _getMinMax4[1];

            // set null data and create new zrange, min, and max
            var _getMinMax5 = this.getMinMax(zrange);

            var _getMinMax6 = _slicedToArray(_getMinMax5, 2);

            zmin = _getMinMax6[0];
            zmax = _getMinMax6[1];
            var data = this.setNull(this.data);
            zrange = data.map(function (o) {
                return o[2];
            });
            zrange = zrange.filter(this.filterRange);
            // [zmin, zmax] = this.getMinMax(zrange);

            var _quantileClip = this.quantileClip(zrange);

            var _quantileClip2 = _slicedToArray(_quantileClip, 2);

            zmin = _quantileClip2[0];
            zmax = _quantileClip2[1];


            if (this.title.toLowerCase().indexOf("vel") >= 0) {
                var cmap = "RdBu";
                // make velocity maps symmetric
                var zabsmax = Math.max.apply(null, [Math.abs(zmin), Math.abs(zmax)]);
                zmin = -zabsmax;
                zmax = zabsmax;
            } else if (this.title.toLowerCase().indexOf("sigma") >= 0) {
                var cmap = "inferno";
            } else {
                var cmap = "linearLab";
            };

            var cstops = this.setColorStops(cmap);

            // make the highcharts
            this.mapdiv.highcharts({
                chart: {
                    type: 'heatmap',
                    marginTop: 40,
                    marginBottom: 80,
                    plotBorderWidth: 1,
                    backgroundColor: null,
                    plotBackgroundColor: '#A8A8A8'
                },
                credits: { enabled: false },
                title: { text: this.title },
                navigation: {
                    buttonOptions: {
                        theme: { fill: null }
                    }
                },
                xAxis: {
                    title: { text: 'Spaxel X' },
                    minorGridLineWidth: 0,
                    min: xymin,
                    max: xymax,
                    tickInterval: 1,
                    tickLength: 0
                },
                yAxis: {
                    title: { text: 'Spaxel Y' },
                    min: xymin,
                    max: xymax,
                    tickInterval: 1,
                    endOnTick: false,
                    gridLineWidth: 0
                },
                colorAxis: {
                    min: zmin,
                    max: zmax,
                    minColor: cstops[0][1],
                    maxColor: cstops[cstops.length - 1][1],
                    stops: cstops,
                    labels: { align: 'center' },
                    reversed: false,
                    startOnTick: false,
                    endOnTick: false,
                    tickPixelInterval: 30,
                    type: "linear"
                },
                plotOptions: {
                    heatmap: {
                        nullColor: 'url(#custom-pattern)' //'#A8A8A8'
                    }
                },
                defs: {
                    patterns: [{
                        width: 3,
                        height: 3,
                        'id': 'custom-pattern',
                        'path': {
                            // I *think* M and L define the start and end points of line segments of the
                            // pattern in units of the width and height, which both default to 10. To
                            // change the density of the pattern hatching, decrease the width and height
                            // and then scale down the "d" values accorindingly.
                            // The second and third set of M and L coordinates color in the upper right
                            // and lower left corners of the box to make the line segments of the
                            // adjacent boxes look continuous. This isn't needed for the vertical or
                            // horizontal hatching.
                            // d: 'M 0 0 L 10 10 M 9 -1 L 11 1 M -1 9 L 1 11',
                            // d: 'M 0 0 L 7 7 M 6 -1 L 8 1 M -1 6 L 1 8',
                            d: 'M 0 0 L 3 3 M 0 3 L 3 0',
                            stroke: 'white', // '#A8A8A8',
                            strokeWidth: 0.3
                        }
                    }]
                },
                legend: {
                    align: 'right',
                    layout: 'vertical',
                    verticalAlign: 'middle',
                    title: { text: this.parameter }
                },
                tooltip: {
                    formatter: function formatter() {
                        return '<br>(' + this.point.x + ', ' + this.point.y + '): <b>' + this.point.value + '</b><br>';
                    }
                },
                series: [{
                    type: "heatmap",
                    data: data,
                    dataLabels: { enabled: false },
                    events: {
                        click: function click(event) {
                            _galthis.getSpaxel(event);
                        }
                    }
                }]
            });
        }
    }]);

    return HeatMap;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-10-20 22:51:56
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Marvin = function () {
    function Marvin(options) {
        _classCallCheck(this, Marvin);

        // set options
        //_.defaults(options, {fruit: "strawberry"})
        this.options = options;

        // set up utility functions
        this.utils = new Utils();
        this.utils.print();
        this.utils.initInfoPopOvers();
        this.utils.initToolTips();

        // load the header
        this.header = new Header();
        this.header.print();

        // setup raven
        this.setupRaven();
    }

    // sets the Sentry raven for monitoring


    _createClass(Marvin, [{
        key: 'setupRaven',
        value: function setupRaven() {
            Raven.config('https://98bc7162624049ffa3d8d9911e373430@sentry.io/107924', {
                release: '0.2.0b1',
                // we highly recommend restricting exceptions to a domain in order to filter out clutter
                whitelistUrls: ['/(sas|api)\.sdss\.org/marvin/', '/(sas|api)\.sdss\.org/marvin2/'],
                includePaths: ['/https?:\/\/((sas|api)\.)?sdss\.org/marvin', '/https?:\/\/((sas|api)\.)?sdss\.org/marvin2']
            }).install();
        }
    }]);

    return Marvin;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 17:38:25
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-26 18:12:18
*/

//
// Javascript object handling all things related to OpenLayers Map
//

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var OLMap = function () {

    // Constructor
    function OLMap(image) {
        _classCallCheck(this, OLMap);

        if (image === undefined) {
            console.error('Must specify an input image to initialize a Map!');
        } else {
            this.image = image;
            this.staticimdiv = $('#staticimage')[0];
            this.mapdiv = $('#imagediv')[0];
            this.getImageSize();
            this.setProjection();
            this.setView();
            this.initMap();
            this.addDrawInteraction();
        }
    }

    _createClass(OLMap, [{
        key: 'print',


        // test print
        value: function print() {
            console.log('We are now printing openlayers map');
        }
    }, {
        key: 'getImageSize',


        // Get the natural size of the input static image
        value: function getImageSize() {
            if (this.staticimdiv !== undefined) {
                this.imwidth = this.staticimdiv.naturalWidth;
                this.imheight = this.staticimdiv.naturalHeight;
            }
        }
    }, {
        key: 'setMouseControl',


        // Set the mouse position control
        value: function setMouseControl() {
            var mousePositionControl = new ol.control.MousePosition({
                coordinateFormat: ol.coordinate.createStringXY(4),
                projection: 'EPSG:4326',
                // comment the following two lines to have the mouse position be placed within the map.
                //className: 'custom-mouse-position',
                //target: document.getElementById('mouse-position'),
                undefinedHTML: '&nbsp;'
            });
            return mousePositionControl;
        }
    }, {
        key: 'setProjection',


        // Set the image Projection
        value: function setProjection() {
            this.extent = [0, 0, this.imwidth, this.imheight];
            this.projection = new ol.proj.Projection({
                code: 'ifu',
                units: 'pixels',
                extent: this.extent
            });
        }
    }, {
        key: 'setBaseImageLayer',


        // Set the base image Layer
        value: function setBaseImageLayer() {
            var imagelayer = new ol.layer.Image({
                source: new ol.source.ImageStatic({
                    url: this.image,
                    projection: this.projection,
                    imageExtent: this.extent
                })
            });
            return imagelayer;
        }
    }, {
        key: 'setView',


        // Set the image View
        value: function setView() {
            this.view = new ol.View({
                projection: this.projection,
                center: ol.extent.getCenter(this.extent),
                zoom: 1,
                maxZoom: 8,
                maxResolution: 1.4
            });
        }
    }, {
        key: 'initMap',


        // Initialize the Map
        value: function initMap() {
            var mousePositionControl = this.setMouseControl();
            var baseimage = this.setBaseImageLayer();
            this.map = new ol.Map({
                controls: ol.control.defaults({
                    attributionOptions: /** @type {olx.control.AttributionOptions} */{
                        collapsible: false
                    }
                }).extend([mousePositionControl]),
                layers: [baseimage],
                target: this.mapdiv,
                view: this.view
            });
        }
    }, {
        key: 'addDrawInteraction',


        // Add a Draw Interaction
        value: function addDrawInteraction() {
            // set up variable for last saved feature & vector source for point
            var lastFeature;
            var drawsource = new ol.source.Vector({ wrapX: false });
            // create new point vectorLayer
            var pointVector = this.newVectorLayer(drawsource);
            // add the layer to the map
            this.map.addLayer(pointVector);

            // New draw event ; default to Point
            var value = 'Point';
            var geometryFunction, maxPoints;
            this.draw = new ol.interaction.Draw({
                source: drawsource,
                type: /** @type {ol.geom.GeometryType} */value,
                geometryFunction: geometryFunction,
                maxPoints: maxPoints
            });

            // On draw end, remove the last saved feature (point)
            this.draw.on('drawend', function (e) {
                if (lastFeature) {
                    drawsource.removeFeature(lastFeature);
                }
                lastFeature = e.feature;
            });

            // add draw interaction onto the map
            this.map.addInteraction(this.draw);
        }
    }, {
        key: 'newVectorLayer',


        // New Vector Layer
        value: function newVectorLayer(source) {
            // default set to Point, but eventually expand this to different vector layer types
            var vector = new ol.layer.Vector({
                source: source,
                style: new ol.style.Style({
                    fill: new ol.style.Fill({
                        color: 'rgba(255, 255, 255, 0.2)'
                    }),
                    stroke: new ol.style.Stroke({
                        color: '#FF0808',
                        width: 2
                    }),
                    image: new ol.style.Circle({
                        radius: 3,
                        fill: new ol.style.Fill({
                            color: '#FF0808'
                        })
                    })
                })
            });
            return vector;
        }
    }]);

    return OLMap;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-05-13 13:26:21
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-14 10:29:12
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Search = function () {

    // Constructor
    function Search() {
        _classCallCheck(this, Search);

        this.searchform = $('#searchform');
        this.typeahead = $('#searchform .typeahead');
        this.returnparams = $('#returnparams');
        this.parambox = $('#parambox');
        this.searchbox = $("#searchbox");
    }

    // Print


    _createClass(Search, [{
        key: 'print',
        value: function print() {
            console.log('I am Search!');
        }

        // Extract

    }, {
        key: 'extractor',
        value: function extractor(input) {
            var regexp = new RegExp('([^,]+)$');
            // parse input for newly typed text
            var result = regexp.exec(input);
            // select last entry after comma
            if (result && result[1]) {
                return result[1].trim();
            }
            return '';
        }

        // Initialize Query Param Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead(typediv, formdiv, url, fxn) {

            var _this = this;
            var typediv = typediv === undefined ? this.typeahead : $(typediv);
            var formdiv = formdiv === undefined ? this.searchform : $(formdiv);
            var typeurl = url === undefined ? Flask.url_for('search_page.getparams') : url;
            var afterfxn = fxn === undefined ? null : fxn;

            function customQueryTokenizer(str) {
                var newstr = str.toString();
                return [_this.extractor(newstr)];
            };

            // create the bloodhound engine
            this.queryparams = new Bloodhound({
                datumTokenizer: Bloodhound.tokenizers.whitespace,
                //queryTokenizer: Bloodhound.tokenizers.whitespace,
                queryTokenizer: customQueryTokenizer,
                prefetch: typeurl,
                remote: {
                    url: typeurl,
                    filter: function filter(qpars) {
                        return qpars;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.queryparams.initialize();

            // init the search typeahead
            typediv.typeahead('destroy');
            typediv.typeahead({
                showHintOnFocus: true,
                items: 'all',
                source: this.queryparams.ttAdapter(),
                updater: function updater(item) {
                    // used to updated the input box with selected option
                    // item = selected item from dropdown
                    var currenttext = this.$element.val();
                    var removedtemptype = currenttext.replace(/[^,]*$/, '');
                    var newtext = removedtemptype + item + ', ';
                    return newtext;
                },
                matcher: function matcher(item) {
                    // used to determined if a query matches an item
                    var tquery = _this.extractor(this.query);
                    if (!tquery) return false;
                    return ~item.toLowerCase().indexOf(tquery.toLowerCase());
                },
                highlighter: function highlighter(item) {
                    // used to highlight autocomplete results ; returns html
                    var oquery = _this.extractor(this.query);
                    var query = oquery.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
                    return item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
                        return '<strong>' + match + '</strong>';
                    });
                }
            });
        }
    }]);

    return Search;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-09 16:52:45
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Table = function () {

    // Constructor
    function Table(tablediv) {
        _classCallCheck(this, Table);

        this.setTable(tablediv);
    }

    // Print


    _createClass(Table, [{
        key: 'print',
        value: function print() {
            console.log('I am Table!');
        }

        // Set the initial Table

    }, {
        key: 'setTable',
        value: function setTable(tablediv) {
            if (tablediv !== undefined) {
                console.log('setting the table');
                this.table = tablediv;
            }
        }

        // initialize a table

    }, {
        key: 'initTable',
        value: function initTable(url, data) {
            this.url = url;

            // if data
            if (data.columns !== null) {
                var cols = this.makeColumns(data.columns);
            }

            // init the Bootstrap table
            this.table.bootstrapTable({
                classes: 'table table-bordered table-condensed table-hover',
                toggle: 'table',
                pagination: true,
                pageSize: 10,
                pageList: '[10, 20, 50]',
                sidePagination: 'server',
                method: 'post',
                contentType: "application/x-www-form-urlencoded",
                data: data.rows,
                totalRows: data.total,
                columns: cols,
                url: url,
                search: true,
                showColumns: true,
                showToggle: true,
                sortName: 'cube.mangaid',
                sortOrder: 'asc',
                formatNoMatches: function formatNoMatches() {
                    return "This table is empty...";
                }
            });
        }

        // make the Table Columns

    }, {
        key: 'makeColumns',
        value: function makeColumns(columns) {
            var cols = [];
            columns.forEach(function (name, index) {
                var colmap = {};
                colmap['field'] = name;
                colmap['title'] = name;
                colmap['sortable'] = true;
                cols.push(colmap);
            });
            return cols;
        }

        // Handle the Bootstrap table JSON response

    }, {
        key: 'handleResponse',
        value: function handleResponse(results) {
            // load the bootstrap table div
            //console.log(this.table, this.table===null, this);
            if (this.table === null) {
                this.setTable();
            }
            this.table = $('#table');
            //console.log('after', this.table, this.table===null, $('#table'));
            // Get new columns
            var cols = results.columns;
            var cols = [];
            results.columns.forEach(function (name, index) {
                var colmap = {};
                colmap['field'] = name;
                colmap['title'] = name;
                colmap['sortable'] = true;
                cols.push(colmap);
            });

            // Load new options
            this.table.bootstrapTable('refreshOptions', { 'columns': cols, 'totalRows': results.total });

            return results;
        }
    }]);

    return Table;
}();
;/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-10-19 13:29:32
*/

// Javascript code for general things

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Utils = function () {

    // Constructor
    function Utils() {
        _classCallCheck(this, Utils);

        // login handlers
        $('#login-user').on('keyup', this, this.submitLogin); // submit login on keypress
        $('#login-pass').on('keyup', this, this.submitLogin); // submit login on keypress
        $('#login-drop').on('hide.bs.dropdown', this, this.resetLogin); //reset login on dropdown hide
    }

    // Print


    _createClass(Utils, [{
        key: 'print',
        value: function print() {
            console.log('I am Utils!');
        }

        // Build a Form

    }, {
        key: 'buildForm',
        value: function buildForm(keys) {
            var args = Array.prototype.slice.call(arguments, 1);
            var form = {};
            keys.forEach(function (key, index) {
                form[key] = args[index];
            });
            return form;
        }

        // Serialize a Form

    }, {
        key: 'serializeForm',
        value: function serializeForm(id) {
            var form = $(id).serializeArray();
            return form;
        }

        // Unique values

    }, {
        key: 'unique',
        value: function unique(data) {
            return new Set(data);
        }

        // Scroll to div

    }, {
        key: 'scrollTo',
        value: function scrollTo(location) {
            if (location !== undefined) {
                var scrolldiv = $(location);
                $('html,body').animate({ scrollTop: scrolldiv.offset().top }, 1500, 'easeInOutExpo');
            } else {
                $('html,body').animate({ scrollTop: 0 }, 1500, 'easeInOutExpo');
            }
        }

        // Initialize Info Pop-Overs

    }, {
        key: 'initInfoPopOvers',
        value: function initInfoPopOvers() {
            $('.infopop [data-toggle="popover"]').popover();
        }
    }, {
        key: 'initToolTips',


        // Initialize tooltips
        value: function initToolTips() {
            $('[data-toggle="tooltip"]').tooltip();
        }
    }, {
        key: 'login',


        // Login function
        value: function login() {
            var form = $('#loginform').serialize();
            var _this = this;

            $.post(Flask.url_for('index_page.login'), form, 'json').done(function (data) {
                if (data.result.status < 0) {
                    // bad submit
                    _this.resetLogin();
                } else {
                    // good submit
                    if (data.result.message !== '') {
                        var stat = data.result.status === 0 ? 'danger' : 'success';
                        var htmlstr = "<div class='alert alert-" + stat + "' role='alert'><h4>" + data.result.message + "</h4></div>";
                        $('#loginmessage').html(htmlstr);
                    }
                    if (data.result.status === 1) {
                        location.reload(true);
                    }
                }
            }).fail(function (data) {
                alert('Bad login attempt');
            });
        }
    }, {
        key: 'resetLogin',


        // Reset Login
        value: function resetLogin() {
            $('#loginform').trigger('reset');
            $('#loginmessage').empty();
        }
    }, {
        key: 'submitLogin',


        // Submit Login on Keyups
        value: function submitLogin(event) {
            var _this = event.data;
            // login
            if (event.keyCode == 13) {
                if ($('#login-user').val() && $('#login-pass').val()) {
                    _this.login();
                }
            }
        }
    }]);

    return Utils;
}();
