/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian
* @Last Modified time: 2016-05-23 13:54:39
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
        this.mapdiv = this.specdiv.find('#mapdiv');
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
                    labels: ['Wavelength','Flux'],
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

    // Initialize Highcharts heatmap
    initHeatmap(myjson, spaxel, maptitle, spectitle, divname, chartWidth){
        console.log('initHeatmap chartWidth', chartWidth);
        var _this = this;
        $(function () {
            var $container = $('#' + divname),
                chart;
            var cubeside = 34;
            $container.highcharts({
                chart: {type: 'heatmap',
                    marginTop: 40,
                    marginBottom: 80,
                    plotBorderWidth: 1,
                    panning: true,
                    panKey: 'shift',
                    zoomType: 'xy',
                    alignticks: false,
                },
                credits: {enabled: false},
                navigation: {
                    buttonOptions: {
                        theme: {fill: null}
                    }
                },
                title: {text: maptitle},
                xAxis: {
                    title: {text: 'Delta RA'},
                    allowDecimals: false,
                    min: 0,
                    max: cubeside - 1,
                    minorGridLineWidth: 0,
                },
                yAxis: {
                    title: {text: 'Delta DEC'},
                    allowDecimals: false,
                    min: 0,
                    max: cubeside - 1,
                    endontick: false,
                },
                colorAxis: {min: 0, max: 30,
                    // minColor: 'rgba(255,255,255,0)',
                    minColor: '#00BFFF',
                    maxColor: '#000080',
                    reversed: false,
                    labels: {align: 'right'}
                },
                legend: {align: 'right',
                    layout: 'vertical',
                    margin: 0,
                    verticalAlign: 'bottom',
                    y: -53,
                    symbolHeight: 380,
                    title: {text: 'Flux'}
                },
                tooltip: {
                    formatter: function () {
                        return '<br>('+ this.point.x + ', ' + this.point.y + '): <b>' + this.point.value + '</b> Halphas <br>';
                    }
                },
                plotOptions:  {
                    series: {
                        events: {
                            click: function (event) {
                                _this.getSpaxelFromMap(event);
                            }
                        }
                    }
                },
                series: [{
                    type: "heatmap",
                    name: "Halphas",
                    borderWidth: 0,
                    data: myjson,
                    dataLabels: {enabled: false},
                }]
            });
            chart = $container.highcharts();
            if (chartWidth) {
                if (chartWidth > 400){
                    var chartWidth = 400;
                    var chartHeight = 400;
                } else {
                    var chartHeight = chartWidth;
                }
                chart.setSize(chartWidth, chartHeight);
            }
            /*$('<button>+</button>').insertBefore($container).click(function () {
                //chartWidth *= 1.1;
                chartWidth = chartHeight;
                chart.setSize(chartWidth, chartHeight);
            });*/
        });
    }

    getSpaxelFromMap(event) {
        var keys = ['plateifu', 'x', 'y'];
        var form = m.utils.buildForm(keys, this.plateifu, event.point.x, event.point.y);
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

    // Retrieves a new Spaxel from the server based on a given mouse position
    getSpaxel(event) {
        var map = event.map;
        var mousecoords = event.coordinate;
        var keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords'];
        var form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth, this.olmap.imheight, mousecoords);
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
    toggleInteract(image, map, spaxel, maptitle, spectitle, chartWidth) {
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
                console.log('image empty');
                this.initOpenLayers(image);
            }
            // load the map if div is empty
            if (mapempty) {
                this.initHeatmap(map, spaxel, maptitle, spectitle, 'mapdiv', chartWidth);
                this.initHeatmap(map, spaxel, maptitle, spectitle, 'mapdiv2', chartWidth);
                this.initHeatmap(map, spaxel, maptitle, spectitle, 'mapdiv3', chartWidth);
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

