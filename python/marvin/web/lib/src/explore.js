/*
 * Filename: explore.js
 * Project: marvin
 * Author: Brian Cherinka
 * Created: Thursday, 9th July 2020 10:09:14 am
 * License: BSD 3-clause "New" or "Revised" License
 * Copyright (c) 2020 Brian Cherinka
 * Last Modified: Friday, 10th July 2020 1:40:00 pm
 * Modified By: Brian Cherinka
 */


//
// Javascript Explore object handling JS things for exploring batch set of galaxies
//
//jshint esversion: 6
'use strict';


class Explore {

    // Constructor
    constructor(targets) {
        this.targets = targets;
        this.explorediv = $('#explorediv');
        this.mapsdiv = this.explorediv.find('#exmaps');
        this.mapdiv = this.explorediv.find('#exmapdiv1');
        this.mapsbtn = $('#getmapbut');

        this.mapparam = $('#mapchoice');
        this.bintemp = $('#btchoice');

        this.mapsbtn.on('click', this, this.testall); // this event fires when a user clicks the MapSpec View Tab

    }

    // Test print
    print() {
        console.log('We are now printing explore', this.targets);
    }

    // Initialize the DAP Heatmap displays
    initHeatmap(maps, mapmsgs) {
        var start = performance.now();
        let mapchildren = this.mapsdiv.children('div');
        let _this = this;
        $.each(mapchildren, function (index, child) {
            let mapdiv = $(child).find('div').first();
            mapdiv.empty();
            if (maps[index] !== undefined && maps[index].data !== null) {
                this.heatmap = new HeatMap(mapdiv, maps[index].data, maps[index].msg,
                    maps[index].plotparams, _this);
                this.heatmap.mapdiv.highcharts().reflow();
            } else {
                let err = `<p class='alert alert-danger'>${mapmsgs[index]}</p>`;
                mapdiv.html(err);
            }
        });
        var end = performance.now();
        console.log('td', end-start, 'in ms');
    }

    // Init a single Heatmap
    initSingleHeatmap(data) {
        let _this = this;
        //let mapdiv = $(child).find('div').first();
        //mapdiv.empty();
        let [mapobj, mapmsg, div] = data;
        let mapdiv = $(div).find('div').first();
        mapdiv.empty();
        if (mapobj !== undefined && mapobj.data !== null) {
            this.heatmap = new HeatMap(mapdiv, mapobj.data, mapobj.msg,
                mapobj.plotparams, _this);
            this.heatmap.mapdiv.highcharts().reflow();
        } else {
            let err = `<p class='alert alert-danger'>${mapmsg}</p>`;
            mapdiv.html(err);
        }
    }

    // Parallel process heatmaps
    parallelHeatmaps(maps, mapmsgs) {
        const zip = (a, b, c) => a.map((k, i) => [k, b[i], c[i]]);
        let mapdivs = this.mapsdiv.children('div');//.find("div");
        let data = zip(maps, mapmsgs, mapdivs);
        console.log(data);
        const p = new Parallel(data);
        p.map(this.initSingleHeatmap);
        //this.initSingleHeatmap(data[0]);
    }

    promise(maps, mapmsgs) {
        this.mapsbtn.prop('disabled', true);
        var start = performance.now();
        let _this = this;
        const zip = (a, b, c) => a.map((k, i) => [k, b[i], c[i]]);
        let mapdivs = this.mapsdiv.children('div');//.find("div");
        let data = zip(maps, mapmsgs, mapdivs);
        
        Promise.all(data.map(function(id) { 
            //console.log('id', id);
            _this.initSingleHeatmap(id);
            return "";
        })).then(function(results) {
            // results is an array of names
            console.log('res', results);
            var end = performance.now();
            console.log('td', end-start, 'in ms');
            _this.mapsbtn.prop('disabled', false);
            _this.mapsbtn.button('reset');
        });
    }

    test(input) {
        let _this = this;
        let param = this.mapparam.prop('value');
        let bintemp = this.bintemp.prop('value');

        let [target, div] = input;
        let mapdiv = $(div).find('div').first();
        mapdiv.empty();

        let data = {'body': JSON.stringify({'target': target, 
            'mapchoice': param, 'btchoice': bintemp}), 
            'method':'POST', 'headers':{'Content-Type':'application/json'}};
        
        //let mapdivs = this.mapsdiv.children('div');
        //let mapdiv = mapdivs.find('div').first();
        //mapdiv.empty();
        //console.log('test', target, mapdiv);

        fetch(Flask.url_for('explore_page.webmap'), data)
        .then((response) => response.json())
        .then((data) => {
            //console.log('json', data.result);
            if (data.result.maps !== undefined && data.result.maps.data !== null) {
                this.heatmap = new HeatMap(mapdiv, data.result.maps.data, data.result.maps.msg,
                    data.result.maps.plotparams, _this);
                this.heatmap.mapdiv.highcharts().reflow();
            } else {
                let err = `<p class='alert alert-danger'>${data.result.msg}</p>`;
                mapdiv.html(err);
            }
        });
        //fetch('http://localhost:5000/marvin/explore/webmap/', {'body': JSON.stringify({'release':'DR17', 'target':'8485-1901', 'mapchoice':'emline_gflux_ha_6564', 'btchoice':'HYB10-MILESHC-MASTARSSP'}), 'method':'POST', 'headers':{'Content-Type':'application/json'}}).then(response=>response.json()).then(data => console.log(data));
    }

    testall(event) {
        let _this = event.data;
        // if (targets === undefined) {
        //     targets = this.targets;
        // }
        let targets = _this.targets;

        var start = performance.now();
        //let _this = this;
        //_this.mapsdiv.children('div').empty();
        const zip = (a, b) => a.map((k, i) => [k, b[i]]);
        let data = zip(targets, _this.mapsdiv.children('div'));
        Promise.all(data.map((d) => {
            //let mapdiv = _this.mapsdiv.children('div').find('div[id^="exmapdiv"]')[index];
            //console.log(mapdiv);
            _this.test(d);
        }
        )).then(response=>{
            var end = performance.now();
            console.log('td', end-start, 'in ms');
            _this.mapsbtn.prop('disabled', false);
            _this.mapsbtn.button('reset');
            console.log("all done");
        });
    }

}