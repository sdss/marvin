/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 16:49:00
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2018-11-17 14:36:53
*/

//
// Javascript Galaxy object handling JS things for a single galaxy
//
//jshint esversion: 6
'use strict';

class SpaxelError extends Error {
  constructor(message) {
    super(message);
    this.message = message;
    this.name = 'SpaxelError';
    this.status = -1;
  }
}

class MapError extends Error {
  constructor(message) {
    super(message);
    this.message = message;
    this.name = 'MapError';
    this.status = -1;
  }
}

class Galaxy {

    // Constructor
    constructor(plateifu, toggleon, redshift) {
        this.setPlateIfu(plateifu);
        this.toggleon = toggleon;
        this.redshift = redshift;
        // main elements
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
        this.maptab = $('#maptab');
        // toggle elements
        this.togglediv = $('#toggleinteract');
        this.toggleload = $('#toggle-load');
        this.togglediv.bootstrapToggle('off');
        this.toggleframe = $('#toggleframe');
        this.togglelines= $('#togglelines'); 
        // flag popover elements
        this.qualpop = $('#qualitypopover');
        this.targpops = $('.targpopovers');
        // maps elements
        this.dapmapsbut = $('#dapmapsbut');
        this.dapselect = $('#dapmapchoices');
        this.dapbt = $('#dapbtchoices');
        //this.dapselect.selectpicker('deselectAll');
        this.resetmapsbut = $('#resetmapsbut');
        // nsa elements
        this.nsadisplay = $('#nsadisp');    // the NSA Display tab element
        this.nsaplots = $('.marvinplot');   // list of divs for the NSA highcharts scatter plot
        this.nsaplotdiv = this.maindiv.find('#nsahighchart1');  // the first div - NSA scatter plot
        this.nsaboxdiv = this.maindiv.find('#nsad3box');  // the NSA D3 boxplot element
        this.nsaselect = $('.nsaselect');//$('#nsachoices1');   // list of the NSA selectpicker elements
        this.nsamsg = this.maindiv.find('#nsamsg');     // the NSA error message element
        this.nsaresetbut = $('.nsareset');//$('#resetnsa1');    // list of the NSA reset button elements
        this.nsamovers = $('#nsatable').find('.mover');     // list of all NSA table parameter name elements
        this.nsaplotbuttons = $('.nsaplotbuts'); // list of the NSA plot button elements
        this.nsatable = $('#nsatable'); // the NSA table element
        this.nsaload = $('#nsa-load'); //the NSA scatter plot loading element

        // object for mapping magnitude bands to their array index
        this.magband = {'F':0, 'N':1, 'u':2, 'g':3, 'r':4, 'i':5, 'z':6};

        // init some stuff
        this.initFlagPopovers();

        //Event Handlers
        this.maptab.on('click', this, this.resizeSpecView); // this event fires when a user clicks the MapSpec View Tab
        this.dapmapsbut.on('click', this, this.getDapMaps); // this event fires when a user clicks the GetMaps button
        this.resetmapsbut.on('click', this, this.resetMaps); // this event fires when a user clicks the Maps Reset button
        this.togglediv.on('change', this, this.initDynamic); // this event fires when a user clicks the Spec/Map View Toggle
        this.nsadisplay.on('click', this, this.displayNSA); // this event fires when a user clicks the NSA tab
        this.nsaresetbut.on('click', this, this.resetNSASelect); // this event fires when a user clicks the NSA select reset button
        //this.nsaselect.on('changed.bs.select', this, this.updateNSAPlot); // this event fires when a user selects an NSA parameter
        this.nsaplotbuttons.on('click', this, this.updateNSAPlot);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        //this.nsatable.on('page-change.bs.table', this, this.updateTableEvents);
        this.toggleframe.on('change', this, this.toggleWavelength); // this event fires when a user clicks the Toggle Obs/Rest Frame
        this.togglelines.on('change', this, this.toggleLines); // this event fires when a user clicks the Toggle Lines

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

    // Resize the ouput MapSpec View when tab clicked
    resizeSpecView(event) {
        let _this = event.data;
        // wait 10 milliseconds before resizing so divs will have the correct size
        m.utils.window[0].setTimeout(function () {
            _this.webspec.resize();
            _this.olmap.map.updateSize();
        }, 10);
    }

    // Compute rest-frame wavelength
    computeRestWave() {
        let rest = this.setSpectrumAxisFormatter('rest', this.redshift);
        this.obswave = this._spaxeldata.map(x => x[0]);
        this.restwave = this.obswave.map(x => rest(x));
        this.rest_spaxeldata = this._spaxeldata.map((x, i) => [this.restwave[i], x.slice(1)[0], x.slice(1)[1]]);
    }

    // Initialize and Load a DyGraph spectrum
    loadSpaxel(spaxel, title) {
        // this plugin renables dygraphs 1.1 behaviour of unzooming to specified valueRange 
        const doubleClickZoomOutPlugin = {
            activate: function (g) {
                // Save the initial y-axis range for later.
                const initialValueRange = g.getOption('valueRange');
                return {
                    dblclick: e => {
                        e.dygraph.updateOptions({
                            dateWindow: null,  // zoom all the way out
                            valueRange: initialValueRange  // zoom to a specific y-axis range.
                        });
                        e.preventDefault();  // prevent the default zoom out action.
                    }
                }
            }
        };

        this.setupSpaxel(spaxel);

        // initialize Dygraph
        let labels = (spaxel[0].length == 3) ? ['Wavelength','Flux', 'Model Fit'] : ['Wavelength','Flux'];
        let options = {
            title: title,
            labels: labels,
            legend: 'always',
            errorBars: true,  // TODO DyGraph shows 2-sigma error bars FIX THIS
            ylabel: 'Flux [10<sup>-17</sup> erg/cm<sup>2</sup>/s/Å]',
            valueRange: [0, null],
            plugins: [doubleClickZoomOutPlugin],
        };
        let data = this.toggleframe.prop('checked') ? this.rest_spaxeldata : spaxel;
        options = this.addDygraphWaveOptions(options);
        this.webspec = new Dygraph(this.graphdiv[0], data, options);

        // create dap line annotations and conditionally display
        this.updateAnnotations();
    }

    // set up some spaxel data arrays
    setupSpaxel(spaxel) {
        this._spaxeldata = spaxel;
        this.computeRestWave();
        this.annotations = this.daplines.map(this.createAnnotation, this);
    } 

    // Dygraph Axis Formatter
    setSpectrumAxisFormatter(wave, redshift) {                
        let obs = (d, gran) => d;
        let rest = (d, gran) => parseFloat((d / (1 + redshift)).toPrecision(5));

        if (wave === 'obs') {
            return obs;            
         } else if (wave === 'rest') {
            return rest;
         }
    }

    addDygraphWaveOptions(oldoptions) {
        let newopts = {};
        if (this.toggleframe.prop('checked')) {
            newopts = { 'file': this.rest_spaxeldata, 'xlabel': 'Rest Wavelength [Ångströms]' };
        } else {
            newopts = { 'file': this._spaxeldata, 'xlabel': 'Observed Wavelength [Ångströms]' };
        }
        let options = Object.assign(oldoptions, newopts);
        return options;         
    }

    // Toggle the Observed/Rest Wavelength
    toggleWavelength(event) {
        let _this = event.data;
        let options = {};
        options = _this.addDygraphWaveOptions(options);
        _this.webspec.updateOptions(options);
        _this.updateAnnotations();
    }

    updateAnnotations() {
        if (this.togglelines.prop('checked')) {
            this.annotations = this.daplines.map(this.createAnnotation, this);
            this.webspec.setAnnotations(this.annotations);
        } else {
            this.webspec.setAnnotations([]);
        }
    }

    // Toggle Line Display
    toggleLines(event) {
        let _this = event.data;
        _this.updateAnnotations();
    }

    // create a Dygraph annotation for measured DAP lines
    createAnnotation(x, index, arr) {
        let [name, wave] = x.split(' ');
        name = name.includes('-') ? name.split('-')[0] + name.split('-')[1][0] : name;
        let owave = parseInt(wave) * (1 + this.redshift);
        let wavedata = (this.toggleframe.prop('checked')) ? this.restwave : this.obswave;
        let use_wave = (this.toggleframe.prop('checked')) ? wave : owave;
        let diff = wavedata.map(y => Math.abs(y - use_wave));
        let idx = diff.indexOf(Math.min(...diff));
        let closest_wave = String(wavedata[idx]);
        let annot = { 'series': "Flux", 'x': closest_wave, 'shortText': name, 'text': x, 'width': 30, 'height': 20, 'tickHeight': 40, 'tickColor': '#cd5c5c', 'cssClass': 'annotation'};
        return annot;
    }

    // Update the spectrum message div for errors only
    updateSpecMsg(specmsg, status) {
        this.specmsg.hide();
        if (status !== undefined && status === -1) {
            this.specmsg.show();
        }
        specmsg = specmsg.replace('<', '').replace('>', '');
        let newmsg = `<strong>${specmsg}</strong>`;
        this.specmsg.empty();
        this.specmsg.html(newmsg);
    }

    // Update a DyGraph spectrum
    updateSpaxel(spaxel, specmsg) {
        this.setupSpaxel(spaxel);
        this.updateSpecMsg(specmsg);
        let options = {'title': specmsg};
        options = this.addDygraphWaveOptions(options);
        this.webspec.updateOptions(options);
        this.updateAnnotations();
    }

    // Initialize OpenLayers Map
    initOpenLayers(image) {
        this.image = image;
        this.olmap = new OLMap(image);
        // add click event handler on map to get spaxel
        this.olmap.map.on('singleclick', this.getSpaxel, this);
    }

    initHeatmap(maps) {
        let mapchildren = this.mapsdiv.children('div');
        let _this = this;
        $.each(mapchildren, function(index, child) {
            let mapdiv = $(child).find('div').first();
            mapdiv.empty();
            if (maps[index] !== undefined && maps[index].data !== null) {
                this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg,
                    maps[index].plotparams, _this);
                this.heatmap.mapdiv.highcharts().reflow();
            }
        });
    }

    // Make Promise error message
    makeError(name) {
        return `Unknown Error: the ${name} javascript Ajax request failed!`;
    }

    // Retrieves a new Spaxel from the server based on a given mouse position or xy spaxel coord.
    getSpaxel(event) {
        let mousecoords = (event.coordinate === undefined) ? null : event.coordinate;
        let divid = $(event.target).parents('div').first().attr('id');
        let maptype = (divid !== undefined && divid.search('highcharts') !== -1) ? 'heatmap' : 'optical';
        let x = (event.point === undefined) ? null : event.point.x;
        let y = (event.point === undefined) ? null : event.point.y;
        let keys = ['plateifu', 'image', 'imwidth', 'imheight', 'mousecoords', 'type', 'x', 'y'];
        let form = m.utils.buildForm(keys, this.plateifu, this.image, this.olmap.imwidth,
            this.olmap.imheight, mousecoords, maptype, x, y);

        // send the form data
        Promise.resolve($.post(Flask.url_for('galaxy_page.getspaxel'), form,'json'))
            .then((data)=>{
                if (data.result.status === -1) {
                    throw new SpaxelError(`Error: ${data.result.specmsg}`);
                }
                this.updateSpaxel(data.result.spectra, data.result.specmsg);
            })
            .catch((error)=>{
                let errmsg = (error.message === undefined) ? this.makeError('getSpaxel') : error.message;
                this.updateSpecMsg(errmsg, -1);
            });
    }

    // check the toggle preference on initial page load
    // eventually for user preferences
    checkToggle() {
        if (this.toggleon === 'true') {
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
        let _this = event.data;
        console.log('toggling', _this.togglediv.prop('checked'), _this.togglediv.hasClass('active'));
    }

    // Initialize the Dynamic Galaxy Interaction upon toggle - makes loading an AJAX request
    initDynamic(event) {

        let _this = event.data;

        if (!_this.togglediv.prop('checked')){
            // Turning Off
            _this.toggleOff();
        } else {
            // Turning On
            _this.toggleOn();

            // check for empty divs
            let specempty = _this.graphdiv.is(':empty');
            let imageempty = _this.imagediv.is(':empty');
            let mapempty = _this.mapdiv.is(':empty');

            // send the request if the dynamic divs are empty
            if (imageempty) {
                // make the form
                let keys = ['plateifu', 'toggleon'];
                let form = m.utils.buildForm(keys, _this.plateifu, _this.toggleon);
                _this.toggleload.show();

                // send the form data
                Promise.resolve($.post(Flask.url_for('galaxy_page.initdynamic'), form,'json'))
                    .then((data)=>{
                        if (data.result.error) {
                            let err = data.result.error;
                            throw new SpaxelError(`Error : ${err}`);
                        }
                        if (data.result.specstatus === -1) {
                            throw new SpaxelError(`Error: ${data.result.specmsg}`);
                        }
                        if (data.result.mapstatus === -1) {
                            throw new MapError(`Error: ${data.result.mapmsg}`);
                        }

                        let image = data.result.image;
                        let spaxel = data.result.spectra;
                        let spectitle = data.result.specmsg;
                        let maps = data.result.maps;
                        let mapmsg = data.result.mapmsg;
                        _this.daplines = data.result.daplines;
                        // Load the Galaxy Image
                        _this.initOpenLayers(image);
                        _this.toggleload.hide();

                        // Load the Spaxel and Maps
                        _this.loadSpaxel(spaxel, spectitle);
                        _this.initHeatmap(maps);
                        // refresh the map selectpicker
                        _this.dapselect.selectpicker('refresh');
                    })
                    .catch((error)=>{
                        let errmsg = (error.message === undefined) ? this.makeError('initDynamic') : error.message;
                        _this.updateSpecMsg(errmsg, -1);
                        _this.updateMapMsg(errmsg, -1);
                    });
            }
        }
    }

    //  Initialize the Quality and Target Popovers
    initFlagPopovers() {
        // DRP Quality Popovers
        this.qualpop.popover({html:true,content:$('#list_drp3quality').html()});
        // MaNGA Target Popovers
        $.each(this.targpops, function(index, value) {
            // get id of flag link
            let popid = value.id;
            // split id and grab the mngtarg
            let [base, targ] = popid.split('_');
            // build the label list id
            let listid = `#list_${targ}`;
            // init the specific popover
            $('#'+popid).popover({html:true,content:$(listid).html()});
        });
    }

    // Get some DAP Maps
    getDapMaps(event) {
        let _this = event.data;
        let params = _this.dapselect.selectpicker('val');
        let bintemp = _this.dapbt.selectpicker('val');
        let keys = ['plateifu', 'params', 'bintemp'];
        let form = m.utils.buildForm(keys, _this.plateifu, params, bintemp);
        _this.mapmsg.hide();
        $(this).button('loading');

        // send the form data
        Promise.resolve($.post(Flask.url_for('galaxy_page.updatemaps'), form, 'json'))
            .then((data)=>{
                if (data.result.status === -1) {
                    throw new MapError(`Error: ${data.result.mapmsg}`);
                }
                _this.dapmapsbut.button('reset');
                _this.initHeatmap(data.result.maps);
            })
            .catch((error)=>{
                let errmsg = (error.message === undefined) ? _this.makeError('getDapMaps') : error.message;
                _this.updateMapMsg(errmsg, -1);
                _this.dapmapsbut.button('reset');
            });
    }

    // Update the Map Msg
    updateMapMsg(mapmsg, status) {
        this.mapmsg.hide();
        if (status !== undefined && status === -1) {
            this.mapmsg.show();
        }
        mapmsg = mapmsg.replace('<', '').replace('>', '');
        let newmsg = `<strong>${mapmsg}</strong>`;
        this.mapmsg.empty();
        this.mapmsg.html(newmsg);
    }

    // Reset the Maps selection
    resetMaps(event) {
        let _this = event.data;
        _this.mapmsg.hide();
        _this.dapselect.selectpicker('deselectAll');
        _this.dapselect.selectpicker('refresh');
    }

    // Set if the galaxy has NSA data or not
    hasNSA(hasnsa) {
        this.hasnsa = hasnsa;
    }

    // Display the NSA info
    displayNSA(event) {
        let _this = event.data;

        // make the form
        let keys = ['plateifu'];
        let form = m.utils.buildForm(keys, _this.plateifu);

        // send the request if the div is empty
        let nsaempty = _this.nsaplots.is(':empty');
        if (nsaempty & _this.hasnsa) {
            // send the form data
            Promise.resolve($.post(Flask.url_for('galaxy_page.initnsaplot'), form, 'json'))
                .then((data)=>{
                    if (data.result.status !== 1) {
                        throw new Error(`Error: ${data.result.nsamsg}`);
                    }
                    _this.addNSAData(data.result.nsa);
                    _this.refreshNSASelect(data.result.nsachoices);
                    _this.initNSAScatter();
                    _this.setTableEvents();
                    _this.addNSAEvents();
                    _this.initNSABoxPlot(data.result.nsaplotcols);
                    _this.nsaload.hide();
                })
                .catch((error)=>{
                    let errmsg = (error.message === undefined) ? _this.makeError('displayNSA') : error.message;
                    _this.updateNSAMsg(errmsg, -1);
                });
        }

    }

    // add the NSA data into the Galaxy object
    addNSAData(data) {
        // the galaxy
        if (data[this.plateifu]) {
            this.mygalaxy = data[this.plateifu];
        } else {
            this.updateNSAMsg(`Error: No NSA data found for ${this.plateifu}`, -1);
            return;
        }
        // the manga sample
        if (data.sample) {
            this.nsasample = data.sample;
        } else {
            this.updateNSAMsg('Error: Problem getting NSA data found for the MaNGA sample', -1);
            return;
        }

    }

    // get new NSA data based on drag-drop axis change
    updateNSAData(index, type) {
        let data, options;
        if (type === 'galaxy') {
            let x = this.mygalaxy[this.nsachoices[index].x];
            let y = this.mygalaxy[this.nsachoices[index].y];
            let pattern = 'absmag_[a-z]$';
            let xrev = (this.nsachoices[index].x.search(pattern) > -1) ? true : false;
            let yrev = (this.nsachoices[index].y.search(pattern) > -1) ? true : false;
            data = [{'name':this.plateifu,'x':x, 'y':y}];
            options = {xtitle:this.nsachoices[index].xtitle, ytitle:this.nsachoices[index].ytitle,
                       title:this.nsachoices[index].title, galaxy:{name:this.plateifu}, xrev:xrev,
                       yrev:yrev};
        } else if (type === 'sample') {
            let x = this.nsasample[this.nsachoices[index].x];
            let y = this.nsasample[this.nsachoices[index].y];
            data = [];
            $.each(x, (index, value)=>{
                if (value > -9999 && y[index] > -9999) {
                    let tmp = {'name':this.nsasample.plateifu[index],'x':value, 'y':y[index]};
                    data.push(tmp);
                }
            });
            options = {xtitle:this.nsachoices[index].xtitle, ytitle:this.nsachoices[index].ytitle,
                       title:this.nsachoices[index].title, altseries:{name:'Sample'}};
        }
        return [data, options];
    }

    // Update the Table event handlers when the table state changes
    setTableEvents() {
        let tabledata = this.nsatable.bootstrapTable('getData');

        $.each(this.nsamovers, (index, mover)=>{
            let id = mover.id;
            $('#'+id).on('dragstart', this, this.dragStart);
            $('#'+id).on('dragover', this, this.dragOver);
            $('#'+id).on('drop', this, this.moverDrop);
        });

        this.nsatable.on('page-change.bs.table', ()=>{
            $.each(tabledata, (index, row)=>{
                let mover = row[0];
                let id = $(mover).attr('id');
                $('#'+id).on('dragstart', this, this.dragStart);
                $('#'+id).on('dragover', this, this.dragOver);
                $('#'+id).on('drop', this, this.moverDrop);
            });
        });
    }

    // Add event handlers to the Highcharts scatter plots
    addNSAEvents() {
        //let _this = this;
        // NSA plot events
        this.nsaplots = $('.marvinplot');
        $.each(this.nsaplots, (index, plot)=>{
            let id = plot.id;
            let highx = $('#'+id).find('.highcharts-xaxis');
            let highy = $('#'+id).find('.highcharts-yaxis');

            highx.on('dragover', this, this.dragOver);
            highx.on('dragenter', this, this.dragEnter);
            highx.on('drop', this, this.dropElement);
            highy.on('dragover', this, this.dragOver);
            highy.on('dragenter', this, this.dragEnter);
            highy.on('drop', this, this.dropElement);
        });
    }

    // Update the NSA Msg
    updateNSAMsg(nsamsg, status) {
        this.nsamsg.hide();
        if (status !== undefined && status === -1) {
            this.nsamsg.show();
        }
        let newmsg = `<strong>${nsamsg}</strong>`;
        this.nsamsg.empty();
        this.nsamsg.html(newmsg);
    }

    // remove values of -9999 from arrays
    filterArray(value) {
        return value !== -9999.0;
    }

    // create the d3 data format
    createD3data() {
        let data = [];
        this.nsaplotcols.forEach((column, index)=>{
            let goodsample = this.nsasample[column].filter(this.filterArray);
            let tmp = {'value':this.mygalaxy[column], 'title':column, 'sample':goodsample};
            data.push(tmp);
        });
        return data;
    }

    // initialize the NSA d3 box and whisker plot
    initNSABoxPlot(cols) {
        // test for undefined columns
        if (cols === undefined && this.nsaplotcols === undefined) {
            console.error('columns for NSA boxplot are undefined');
        } else {
            this.nsaplotcols = cols;
        }

        // generate the data format
        let data, options;
        data = this.createD3data();
        this.nsad3box = new BoxWhisker(this.nsaboxdiv, data, options);

    }

    // Destroy old Charts
    destroyChart(div, index) {
        this.nsascatter[index].chart.destroy();
        div.empty();
    }

    // Init the NSA Scatter plot
    initNSAScatter(parentid) {
        // only update the single parent div element
        if (parentid !== undefined) {
            let parentdiv = this.maindiv.find('#'+parentid);
            let index = parseInt(parentid[parentid.length-1]);
            let [data, options] = this.updateNSAData(index, 'galaxy');
            let [sdata, soptions] = this.updateNSAData(index, 'sample');
            options.altseries = {data:sdata, name:'Sample'};
            this.destroyChart(parentdiv, index);
            this.nsascatter[index] = new Scatter(parentdiv, data, options);
        } else {
            // try updating all of them
            this.nsascatter = {};
            $.each(this.nsaplots, (index, plot)=>{
                let plotdiv = $(plot);
                let [data, options] = this.updateNSAData(index+1, 'galaxy');
                let [sdata, soptions] = this.updateNSAData(index+1, 'sample');
                options.altseries = {data:sdata,name:'Sample'};
                this.nsascatter[index+1] = new Scatter(plotdiv, data, options);
            });
        }

    }

    // Refresh the NSA select choices for the scatter plot
    refreshNSASelect(vals) {
        this.nsachoices = vals;
        $.each(this.nsaselect, function(index, nsasp) {
            $(nsasp).selectpicker('deselectAll');
            $(nsasp).selectpicker('val', ['x_'+vals[index+1].x, 'y_'+vals[index+1].y]);
            $(nsasp).selectpicker('refresh');
        });
    }

    // Update the NSA selectpicker choices for the scatter plot
    updateNSAChoices(index, params) {
        let xpar = params[0].slice(2,params[0].length);
        let ypar = params[1].slice(2,params[1].length);
        this.nsachoices[index].title = ypar+' vs '+xpar;
        this.nsachoices[index].xtitle = xpar;
        this.nsachoices[index].x = xpar;
        this.nsachoices[index].ytitle = ypar;
        this.nsachoices[index].y = ypar;
    }

    // Reset the NSA selecpicker
    resetNSASelect(event) {
        let resetid = $(this).attr('id');
        let index = parseInt(resetid[resetid.length-1]);
        let _this = event.data;
        let myselect = _this.nsaselect[index-1];
        _this.nsamsg.hide();
        $(myselect).selectpicker('deselectAll');
        $(myselect).selectpicker('refresh');
    }

    // Update the NSA scatter plot on select change
    updateNSAPlot(event) {
        let _this = event.data;
        let plotid = $(this).attr('id');
        let index = parseInt(plotid[plotid.length-1]);
        let nsasp = _this.nsaselect[index-1];
        let params = $(nsasp).selectpicker('val');

        // Construct the new NSA data
        let parentid = `nsahighchart${index}`;
        _this.updateNSAChoices(index, params);
        _this.initNSAScatter(parentid);
        _this.addNSAEvents();

    }

    // Events for Drag and Drop

    // Element drag start
    dragStart(event) {
        let _this = event.data;
        let param = this.id+'+'+this.textContent;
        event.originalEvent.dataTransfer.setData('Text', param);

        // show the overlay elements
        $.each(_this.nsascatter, (index, scat)=>{ scat.overgroup.show(); });
    }
    // Element drag over
    dragOver(event) {
        event.preventDefault();
        //event.stopPropagation();
        event.originalEvent.dataTransfer.dropEffect = 'move';
    }
    // Element drag enter
    dragEnter(event) {
        event.preventDefault();
        //event.stopPropagation();
    }
    // Mover element drop event
    moverDrop(event) {
        event.preventDefault();
        event.stopPropagation();
    }
    // Element drop and redraw the scatter plot
    dropElement(event) {
        event.preventDefault();
        event.stopPropagation();
        // get the id and name of the dropped parameter
        let _this = event.data;
        let param = event.originalEvent.dataTransfer.getData('Text');
        let [id, name] = param.split('+');

        // Hide overlay elements
        $.each(_this.nsascatter, (index, scat)=>{ scat.overgroup.hide(); });

        // Determine which axis and plot the name was dropped on
        let classes = $(this).attr('class');
        let isX = classes.includes('highcharts-xaxis');
        let isY = classes.includes('highcharts-yaxis');
        let parentdiv = $(this).closest('.marvinplot');
        let parentid = parentdiv.attr('id');
        if (parentid === undefined ){
            event.stopPropagation();
            return false;
        }
        let parentindex = parseInt(parentid[parentid.length-1]);

        // get the other axis and extract title
        let otheraxis = null;
        if (isX) {
            otheraxis = $(this).next();
        } else if (isY) {
            otheraxis = $(this).prev();
        }
        let axistitle = this.textContent;
        let otheraxistitle = otheraxis[0].textContent;

        // Update the Values
        let newtitle = _this.nsachoices[parentindex].title.replace(axistitle, name);
        _this.nsachoices[parentindex].title = newtitle;
        if (isX) {
            _this.nsachoices[parentindex].xtitle = name;
            _this.nsachoices[parentindex].x = id;
        } else if (isY) {
            _this.nsachoices[parentindex].ytitle = name;
            _this.nsachoices[parentindex].y = id;
        }

        // Construct the new NSA data
        _this.initNSAScatter(parentid);
        _this.addNSAEvents();

        return false;

    }

}
