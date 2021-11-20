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
        });
    }

}