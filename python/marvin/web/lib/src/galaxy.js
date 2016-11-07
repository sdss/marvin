/*
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

class Galaxy {

    // Constructor
    constructor(plateifu, toggleon) {
        this.setPlateIfu(plateifu);
        this.toggleon = toggleon;
        this.maindiv = $('#'+this.plateifu);
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
    print() {
        console.log('We are now printing galaxy', this.plateifu, this.plate, this.ifu);
    }

    // Set the plateifu
    setPlateIfu(plateifu) {
        if (plateifu === undefined) {
            this.plateifu = $('.singlegalaxy').attr('id');
        } else {
            this.plateifu = plateifu;
        }
        [this.plate, this.ifu] = this.plateifu.split('-');
    }

    // Initialize and Load a DyGraph spectrum
    loadSpaxel(spaxel, title) {
        var labels = (spaxel[0].length == 3) ? ['Wavelength','Flux', 'Model Fit'] : ['Wavelength','Flux'];
        this.webspec = new Dygraph(this.graphdiv[0],
                  spaxel,
                  {
                    title: title,
                    labels: labels,
                    errorBars: true,  // TODO DyGraph shows 2-sigma error bars FIX THIS
                    ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
                    xlabel: 'Wavelength [Ångströms]'
                  });
    };

    // Update the spectrum message div for errors only
    updateSpecMsg(specmsg, status) {
        this.specmsg.hide();
        if (status !== undefined && status === -1) {
            this.specmsg.show();
        }
        var newmsg = '<strong>'+specmsg+'</strong>';
        this.specmsg.empty();
        this.specmsg.html(newmsg);
    }

    // Update a DyGraph spectrum
    updateSpaxel(spaxel, specmsg) {
        this.updateSpecMsg(specmsg);
        this.webspec.updateOptions({'file': spaxel, 'title':specmsg});
    };

    // Initialize OpenLayers Map
    initOpenLayers(image) {
        this.image = image;
        this.olmap = new OLMap(image);
        // add click event handler on map to get spaxel
        this.olmap.map.on('singleclick', this.getSpaxel, this);
    };

    initHeatmap(maps) {
        console.log('initHeatmap', this.mapsdiv);
        var mapchildren = this.mapsdiv.children('div');
        console.log('mapchildren', mapchildren);
        var _this = this;
        $.each(mapchildren, function(index, child) {
            var mapdiv = $(child).find('div').first();
            mapdiv.empty();
            if (maps[index] !== undefined && maps[index].data !== null) {
                this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg, _this);
                this.heatmap.mapdiv.highcharts().reflow();
            }
        });
    };

    // Retrieves a new Spaxel from the server based on a given mouse position or xy spaxel coord.
    getSpaxel(event) {
        var mousecoords = (event.coordinate === undefined) ? null : event.coordinate;
        var divid = $(event.target).parents('div').first().attr('id');
        var maptype = (divid !== undefined && divid.search('highcharts') !== -1) ? 'heatmap' : 'optical';
        var x = (event.point === undefined) ? null : event.point.x;
        var y = (event.point === undefined) ? null : event.point.y;
        var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords', 'type', 'x', 'y'];
        var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth,
            this.olmap.imheight, mousecoords, maptype, x, y);
        var _this = this;

        // send the form data
        $.post(Flask.url_for('galaxy_page.getspaxel'), form,'json')
            .done(function(data) {
                if (data.result.status !== -1) {
                    _this.updateSpaxel(data.result.spectra, data.result.specmsg);
                } else {
                    _this.updateSpecMsg('Error: '+data.result.specmsg, data.result.status);
                }
            })
            .fail(function(data) {
                _this.updateSpecMsg('Error: '+data.result.specmsg, data.result.status);
            });
    };

    // check the toggle preference on initial page load
    // eventually for user preferences
    checkToggle() {
        if (this.toggleon) {
            this.toggleOn();
        } else {
            this.toggleOff();
        }
    }

    // toggle the display button on
    toggleOn() {
        // eventually this should include the ajax stuff inside initDynamic - for after user preferences implemented
        this.toggleon = true;
        //this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
        //this.togglediv.button('complete');
        this.staticdiv.hide();
        this.dynamicdiv.show();
    }

    // toggle the display button off
    toggleOff() {
        this.toggleon = false;
        //this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
        //this.togglediv.button('reset');
        this.dynamicdiv.hide();
        this.staticdiv.show();
    }

    testTogg(event) {
        var _this = event.data;
        console.log('toggling', _this.togglediv.prop('checked'), _this.togglediv.hasClass('active'));
    }

    // Initialize the Dynamic Galaxy Interaction upon toggle - makes loading an AJAX request
    initDynamic(event) {

        var _this = event.data;

        if (!_this.togglediv.prop('checked')){
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

                $.post(Flask.url_for('galaxy_page.initdynamic'), form, 'json')
                    .done(function(data) {

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
                            _this.updateSpecMsg('Error: '+spectitle, data.result.specstatus);
                        }

                        // Try to load the Maps
                        if (data.result.mapstatus !== -1) {
                            _this.initHeatmap(maps);
                        } else {
                            _this.updateMapMsg('Error: '+mapmsg, data.result.mapstatus);
                        }

                    })
                    .fail(function(data) {
                        _this.updateSpecMsg('Error: '+data.result.specmsg, data.result.specstatus);
                        _this.updateMapMsg('Error: '+data.result.mapmsg, data.result.mapstatus);
                        _this.toggleload.hide();
                    });
            }
        }
    }

    // Toggle the interactive OpenLayers map and Dygraph spectra
    // DEPRECATED - REMOVE
    toggleInteract(image, maps, spaxel, spectitle, mapmsg) {
        if (this.togglediv.hasClass('active')){
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
    };

    //  Initialize the Quality and Target Popovers
    initFlagPopovers() {
        // DRP Quality Popovers
        this.qualpop.popover({html:true,content:$('#list_drp3quality').html()});
        // MaNGA Target Popovers
        $.each(this.targpops, function(index, value) {
            // get id of flag link
            var popid = value.id;
            // split id and grab the mngtarg
            var [base, targ] = popid.split('_');
            // build the label list id
            var listid = '#list_'+targ;
            // init the specific popover
            $('#'+popid).popover({html:true,content:$(listid).html()});
        });
    };

    // Get some DAP Maps
    getDapMaps(event) {
        var _this = event.data;
        console.log('getting dap maps', _this.dapselect.selectpicker('val'));
        var params = _this.dapselect.selectpicker('val');
        var bintemp = _this.dapbt.selectpicker('val');
        var keys = ['plateifu', 'params', 'bintemp'];
        var form = m.utils.buildForm(keys, _this.plateifu, params, bintemp);
        _this.mapmsg.hide();
        $(this).button('loading');

        // send the form data
        $.post(Flask.url_for('galaxy_page.updatemaps'), form, 'json')
            .done(function(data) {
                if (data.result.status !== -1) {
                    _this.dapmapsbut.button('reset');
                    _this.initHeatmap(data.result.maps);
                } else {
                    _this.updateMapMsg('Error: '+data.result.mapmsg, data.result.status);
                    _this.dapmapsbut.button('reset');
                }
            })
            .fail(function(data) {
                _this.updateMapMsg('Error: '+data.result.mapmsg, data.result.status);
                _this.dapmapsbut.button('reset');
            });
    };

    // Update the Map Msg
    updateMapMsg(mapmsg, status) {
        this.mapmsg.hide();
        if (status !== undefined && status === -1) {
            this.mapmsg.show();
        }
        var newmsg = '<strong>'+mapmsg+'</strong>';
        this.mapmsg.empty();
        this.mapmsg.html(newmsg);
    };

    // Reset the Maps selection
    resetMaps(event) {
        var _this = event.data;
        _this.mapmsg.hide();
        _this.dapselect.selectpicker('deselectAll');
        _this.dapselect.selectpicker('refresh');
    }
}
