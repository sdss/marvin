/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-23 01:41:10
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//

'use strict';

class Galaxy {

    // Constructor
    constructor(plateifu) {
        this.setPlateIfu(plateifu);
        this.maindiv = $('#'+this.plateifu);
        this.metadiv = this.maindiv.find('#metadata');
        this.specdiv = this.maindiv.find('#specview');
        this.imagediv = this.specdiv.find('#imagediv');
        this.mapsdiv = this.specdiv.find('#mapsdiv');
        this.mapdiv = this.specdiv.find('#mapdiv1');
        this.graphdiv = this.specdiv.find('#graphdiv');
        this.specmsg = this.specdiv.find('#specmsg');
        this.webspec = null;
        this.staticdiv = this.specdiv.find('#staticdiv');
        this.dynamicdiv = this.specdiv.find('#dynamicdiv');
        this.togglediv = $('#toggleinteract');
        this.qualpop = $('#qualitypopover');
        this.targpops = $('.targpopovers');

        // init some stuff
        this.initFlagPopovers();
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
        this.webspec = new Dygraph(this.graphdiv[0],
                  spaxel,
                  {
                    title: title,
                    labels: ['Wavelength','Flux', 'Flux2'],
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
            this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg, _this);
            this.heatmap.mapdiv.highcharts().reflow();
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

    // Toggle the interactive OpenLayers map and Dygraph spectra
    toggleInteract(image, maps, spaxel, spectitle) {
        if (this.togglediv.hasClass('active')){
            // Turning Off
            this.togglediv.toggleClass('btn-danger').toggleClass('btn-success');
            this.togglediv.button('reset');
            this.dynamicdiv.hide();
            this.staticdiv.show();
        } else {
            // Turning On
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
}
