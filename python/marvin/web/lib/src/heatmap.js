/*
* @Author: Brian Cherinka
* @Date:   2016-08-30 11:28:26
 * @Last modified by:   andrews
 * @Last modified time: 2017-12-13 23:12:53
*/

//jshint esversion: 6
'use strict';

class HeatMap {

    // Constructor
    constructor(mapdiv, data, title, plotparams, galthis) {
        if (data === undefined) {
            console.error('Must specify input map data to initialize a HeatMap!');
        } else if (mapdiv === undefined) {
            console.error('Must specify an input mapdiv to initialize a HeatMap');
        } else {
            this.mapdiv = mapdiv; // div element for map
            this.data = data; // map data
            this.title = title; // map title
            this.plotparams = plotparams; // default plotting parameters
            this.galthis = galthis; //the self of the Galaxy class
            this.parseTitle();
            this.initMap();
            this.setColorNoData(this, Highcharts);
        }

    }

    // test print
    print() {
        console.log('We are now printing heatmap for ', this.title);
    }

    // Parse the heatmap title into category, parameter, channel
    // e.g. 7443-1901: emline_gflux_ha-6564
    parseTitle() {
        let [plateifu, newtitle] = this.title.split(':');
        [this.category, this.parameter, this.channel] = newtitle.split('_');
    }

    // Get range of x (or y) data and z (DAP property) data
    getRange(){
        let xylength  = this.data.values.length;
        let xyrange = Array.apply(null, {length: xylength}).map(Number.call, Number);
        let zrange = [].concat.apply([], this.data.values);
        return [xyrange, zrange];
    }

    // Filter out null and no-data from z (DAP prop) data
    filterRange(z) {
        if (z !== undefined && typeof(z) === 'number' && !isNaN(z)) {
            return true;
        } else {
            return false;
        }
    }

    // return the min and max of a range
    getMinMax(range) {
        // var range = (range === undefined) ? this.getRange() : range;
        let min = Math.min.apply(null, range);
        let max = Math.max.apply(null, range);
        return [min, max];
    }

    setNull(x) {
        let values = x.values;
        let ivar = x.ivar;
        let mask = x.mask;

        let xyz = Array();

        for (let ii=0; ii < values.length; ii++) {
            for (let jj=0; jj < values.length; jj++){
                let val = values[ii][jj];
                let noData, badData;
                let signalToNoise, signalToNoiseThreshold;

                if (mask !== null) {
                    let bitmasks = this.plotparams["bits"];
                    noData = (mask[ii][jj] & Math.pow(2, bitmasks["nocov"]));
                    badData = false;
                    for (let key in bitmasks["badData"]) {
                        badData = badData || (mask[ii][jj] & Math.pow(2, bitmasks["badData"][key]))
                    }
                } else {
                    noData = null;
                    badData = null;
                }
                signalToNoiseThreshold = this.plotparams["snr_min"];
                if (ivar !== null) {
                    signalToNoise = Math.abs(val) * Math.sqrt(ivar[ii][jj]);
                }

                // value types
                // val=no-data => gray color
                // val=null => hatch area

                if (noData) {
                    // for data that is outside the range "NOCOV" mask
                    val = 'no-data';
                } else if (badData) {
                    // for data that is bad - masked in some way
                    val = null;
                } else if (ivar !== null && (signalToNoise < signalToNoiseThreshold)) {
                    // for data that is low S/N
                    val = null ;
                } else if (ivar === null) {
                    // for data with no mask or no inverse variance extensions
                    if (this.title.search('binid') !== -1) {
                        // for binid extension only, set -1 values to no data
                        val = (val == -1 ) ? 'no-data' : val;
                    } else if (val === 0.0) {
                        // set zero values to no-data
                        val = 'no-data';
                    }
                }
                // need to push as jj, ii since the numpy 2-d arrays are y, x based (row, col)
                xyz.push([jj, ii, val]);
            }
        }
        return xyz;
    }

    setColorNoData(_this, H) {
        H.wrap(H.ColorAxis.prototype, 'toColor', function (proceed, value, point) {
            if (value === 'no-data') {
                // make gray color
                return 'rgba(0,0,0,0)';  // '#A8A8A8';
            }
            else if (value === 'low-sn') {
                // make light blue with half-opacity == muddy blue-gray
                return 'rgba(0,191,255,0.5)'; //'#7fffd4';
            }
            else
                return proceed.apply(this, Array.prototype.slice.call(arguments, 1));
        });
    }

    setColorMapHex(cmap){

        let linearLabHex = ['#040404', '#0a0308', '#0d040b', '#10050e', '#120510', '#150612',
        '#160713', '#180815', '#1a0816', '#1b0918', '#1c0a19', '#1e0b1a', '#1f0c1b', '#200c1c',
        '#210d1d', '#230e1f', '#240e20', '#250f20', '#260f21', '#271022', '#281123', '#291124',
        '#2a1226', '#2b1326', '#2c1327', '#2e1429', '#2e142d', '#2e1532', '#2d1537', '#2d153c',
        '#2d1640', '#2d1743', '#2d1747', '#2d184b', '#2d184d', '#2d1951', '#2d1954', '#2c1a57',
        '#2c1b5a', '#2d1b5c', '#2d1c5f', '#2c1d62', '#2c1d64', '#2c1e67', '#2c1f6a', '#2c1f6d',
        '#2c206e', '#2c2171', '#2c2274', '#2c2276', '#2a2379', '#282678', '#262877', '#242a78',
        '#222c78', '#212e78', '#202f78', '#1f3179', '#1e327a', '#1e337b', '#1d347b', '#1d357d',
        '#1c377d', '#1c387e', '#1b397f', '#1c3a80', '#1c3b81', '#1b3c81', '#1b3d83', '#1b3e84',
        '#1b3f85', '#1c4086', '#1b4187', '#1b4288', '#1b4489', '#1b458a', '#194788', '#164986',
        '#154a85', '#144c83', '#114e81', '#104f80', '#0f517e', '#0e527d', '#0a547b', '#0a557a',
        '#095778', '#085877', '#075976', '#065b75', '#045c73', '#045e72', '#045f72', '#036070',
        '#01626f', '#01636e', '#00646d', '#00656c', '#00676b', '#00686a', '#006969', '#006b68',
        '#006c65', '#006e64', '#006f63', '#007062', '#007260', '#00735f', '#00745d', '#00765c',
        '#00775a', '#007859', '#007958', '#007b56', '#007c55', '#007d53', '#007f52', '#008050',
        '#00814f', '#00834d', '#00844b', '#008549', '#008648', '#008846', '#008944', '#008a42',
        '#008b41', '#008d40', '#008e3f', '#008f3d', '#00913c', '#00923c', '#00933a', '#009539',
        '#009638', '#009737', '#009935', '#009a34', '#009b33', '#009d32', '#009e30', '#009f2f',
        '#00a02d', '#00a22c', '#00a32a', '#00a429', '#00a527', '#00a724', '#00a822', '#00a91f',
        '#00aa17', '#00a908', '#09aa00', '#14ab00', '#1dac00', '#23ad00', '#28ae00', '#2daf00',
        '#30b000', '#34b100', '#37b200', '#3bb300', '#3db400', '#40b500', '#42b600', '#44b700',
        '#47b800', '#49b900', '#4cba00', '#4ebb00', '#4fbc00', '#51bd00', '#53be00', '#55bf00',
        '#57c000', '#5cc000', '#63c100', '#6ac100', '#72c100', '#77c200', '#7dc200', '#82c200',
        '#87c300', '#8cc300', '#91c300', '#95c400', '#99c400', '#9dc500', '#a1c500', '#a5c500',
        '#a9c600', '#acc600', '#b0c700', '#b4c700', '#b8c700', '#bac800', '#bec900', '#c1c900',
        '#c5c900', '#c8ca00', '#c9c918', '#cbca33', '#ceca41', '#cfcb4d', '#d1cb57', '#d4cb5f',
        '#d5cc67', '#d7cd6d', '#dacd74', '#dbce79', '#ddcf7f', '#dfcf84', '#e2cf8a', '#e3d08f',
        '#e5d193', '#e7d197', '#e8d29b', '#ebd39f', '#edd3a4', '#eed4a8', '#f0d4ac', '#f3d5af',
        '#f3d6b3', '#f5d6b7', '#f8d7ba', '#f8d8bd', '#f8dac1', '#f7dbc3', '#f7dcc6', '#f7dec9',
        '#f8dfcc', '#f7e0ce', '#f7e2d1', '#f7e3d3', '#f7e5d6', '#f7e6d8', '#f7e7da', '#f7e8dc',
        '#f8eae0', '#f7ebe1', '#f7ece5', '#f7eee7', '#f7efe8', '#f8f0eb', '#f8f2ed', '#f7f3ef',
        '#f8f4f1', '#f8f6f4', '#f8f7f6', '#f8f8f8', '#f9f9f9', '#fbfbfb', '#fcfcfc', '#fdfdfd',
        '#fefefe', '#ffffff'];

        let infernoHex = ['#000004', '#010005',  '#010106',  '#010108',  '#02010a',  '#02020c',
        '#02020e',  '#030210',  '#040312',  '#040314',  '#050417',  '#060419',  '#07051b',
        '#08051d',  '#09061f',  '#0a0722',  '#0b0724',  '#0c0826',  '#0d0829',  '#0e092b',
        '#10092d',  '#110a30',  '#120a32',  '#140b34',  '#150b37',  '#160b39',  '#180c3c',
        '#190c3e',  '#1b0c41',  '#1c0c43',  '#1e0c45',  '#1f0c48',  '#210c4a',  '#230c4c',
        '#240c4f',  '#260c51',  '#280b53',  '#290b55',  '#2b0b57',  '#2d0b59',  '#2f0a5b',
        '#310a5c',  '#320a5e',  '#340a5f',  '#360961',  '#380962',  '#390963',  '#3b0964',
        '#3d0965',  '#3e0966',  '#400a67',  '#420a68',  '#440a68',  '#450a69',  '#470b6a',
        '#490b6a',  '#4a0c6b',  '#4c0c6b',  '#4d0d6c',  '#4f0d6c',  '#510e6c',  '#520e6d',
        '#540f6d',  '#550f6d',  '#57106e',  '#59106e',  '#5a116e',  '#5c126e',  '#5d126e',
        '#5f136e',  '#61136e',  '#62146e',  '#64156e',  '#65156e',  '#67166e',  '#69166e',
        '#6a176e',  '#6c186e',  '#6d186e',  '#6f196e',  '#71196e',  '#721a6e',  '#741a6e',
        '#751b6e',  '#771c6d',  '#781c6d',  '#7a1d6d',  '#7c1d6d',  '#7d1e6d',  '#7f1e6c',
        '#801f6c',  '#82206c',  '#84206b',  '#85216b',  '#87216b',  '#88226a',  '#8a226a',
        '#8c2369',  '#8d2369',  '#8f2469',  '#902568',  '#922568',  '#932667',  '#952667',
        '#972766',  '#982766',  '#9a2865',  '#9b2964',  '#9d2964',  '#9f2a63',  '#a02a63',
        '#a22b62',  '#a32c61',  '#a52c60',  '#a62d60',  '#a82e5f',  '#a92e5e',  '#ab2f5e',
        '#ad305d',  '#ae305c',  '#b0315b',  '#b1325a',  '#b3325a',  '#b43359',  '#b63458',
        '#b73557',  '#b93556',  '#ba3655',  '#bc3754',  '#bd3853',  '#bf3952',  '#c03a51',
        '#c13a50',  '#c33b4f',  '#c43c4e',  '#c63d4d',  '#c73e4c',  '#c83f4b',  '#ca404a',
        '#cb4149',  '#cc4248',  '#ce4347',  '#cf4446',  '#d04545',  '#d24644',  '#d34743',
        '#d44842',  '#d54a41',  '#d74b3f',  '#d84c3e',  '#d94d3d',  '#da4e3c',  '#db503b',
        '#dd513a',  '#de5238',  '#df5337',  '#e05536',  '#e15635',  '#e25734',  '#e35933',
        '#e45a31',  '#e55c30',  '#e65d2f',  '#e75e2e',  '#e8602d',  '#e9612b',  '#ea632a',
        '#eb6429',  '#eb6628',  '#ec6726',  '#ed6925',  '#ee6a24',  '#ef6c23',  '#ef6e21',
        '#f06f20',  '#f1711f',  '#f1731d',  '#f2741c',  '#f3761b',  '#f37819',  '#f47918',
        '#f57b17',  '#f57d15',  '#f67e14',  '#f68013',  '#f78212',  '#f78410',  '#f8850f',
        '#f8870e',  '#f8890c',  '#f98b0b',  '#f98c0a',  '#f98e09',  '#fa9008',  '#fa9207',
        '#fa9407',  '#fb9606',  '#fb9706',  '#fb9906',  '#fb9b06',  '#fb9d07',  '#fc9f07',
        '#fca108',  '#fca309',  '#fca50a',  '#fca60c',  '#fca80d',  '#fcaa0f',  '#fcac11',
        '#fcae12',  '#fcb014',  '#fcb216',  '#fcb418',  '#fbb61a',  '#fbb81d',  '#fbba1f',
        '#fbbc21',  '#fbbe23',  '#fac026',  '#fac228',  '#fac42a',  '#fac62d',  '#f9c72f',
        '#f9c932',  '#f9cb35',  '#f8cd37',  '#f8cf3a',  '#f7d13d',  '#f7d340',  '#f6d543',
        '#f6d746',  '#f5d949',  '#f5db4c',  '#f4dd4f',  '#f4df53',  '#f4e156',  '#f3e35a',
        '#f3e55d',  '#f2e661',  '#f2e865',  '#f2ea69',  '#f1ec6d',  '#f1ed71',  '#f1ef75',
        '#f1f179',  '#f2f27d',  '#f2f482',  '#f3f586',  '#f3f68a',  '#f4f88e',  '#f5f992',
        '#f6fa96',  '#f8fb9a',  '#f9fc9d',  '#fafda1',  '#fcffa4'];

        let RdBuHex = ['#053061', '#063264', '#073467', '#08366a', '#09386d', '#0a3b70',
        '#0c3d73', '#0d3f76', '#0e4179', '#0f437b', '#10457e', '#114781', '#124984', '#134c87',
        '#144e8a', '#15508d', '#175290', '#185493', '#195696', '#1a5899', '#1b5a9c', '#1c5c9f',
        '#1d5fa2', '#1e61a5', '#1f63a8', '#2065ab', '#2267ac', '#2369ad', '#246aae', '#266caf',
        '#276eb0', '#2870b1', '#2a71b2', '#2b73b3', '#2c75b4', '#2e77b5', '#2f79b5', '#307ab6',
        '#327cb7', '#337eb8', '#3480b9', '#3681ba', '#3783bb', '#3885bc', '#3a87bd', '#3b88be',
        '#3c8abe', '#3e8cbf', '#3f8ec0', '#408fc1', '#4291c2', '#4393c3', '#4695c4', '#4997c5',
        '#4c99c6', '#4f9bc7', '#529dc8', '#569fc9', '#59a1ca', '#5ca3cb', '#5fa5cd', '#62a7ce',
        '#65a9cf', '#68abd0', '#6bacd1', '#6eaed2', '#71b0d3', '#75b2d4', '#78b4d5', '#7bb6d6',
        '#7eb8d7', '#81bad8', '#84bcd9', '#87beda', '#8ac0db', '#8dc2dc', '#90c4dd', '#93c6de',
        '#96c7df', '#98c8e0', '#9bc9e0', '#9dcbe1', '#a0cce2', '#a2cde3', '#a5cee3', '#a7d0e4',
        '#a9d1e5', '#acd2e5', '#aed3e6', '#b1d5e7', '#b3d6e8', '#b6d7e8', '#b8d8e9', '#bbdaea',
        '#bddbea', '#c0dceb', '#c2ddec', '#c5dfec', '#c7e0ed', '#cae1ee', '#cce2ef', '#cfe4ef',
        '#d1e5f0', '#d2e6f0', '#d4e6f1', '#d5e7f1', '#d7e8f1', '#d8e9f1', '#dae9f2', '#dbeaf2',
        '#ddebf2', '#deebf2', '#e0ecf3', '#e1edf3', '#e3edf3', '#e4eef4', '#e6eff4', '#e7f0f4',
        '#e9f0f4', '#eaf1f5', '#ecf2f5', '#edf2f5', '#eff3f5', '#f0f4f6', '#f2f5f6', '#f3f5f6',
        '#f5f6f7', '#f6f7f7', '#f7f6f6', '#f7f5f4', '#f8f4f2', '#f8f3f0', '#f8f2ef', '#f8f1ed',
        '#f9f0eb', '#f9efe9', '#f9eee7', '#f9ede5', '#f9ebe3', '#faeae1', '#fae9df', '#fae8de',
        '#fae7dc', '#fbe6da', '#fbe5d8', '#fbe4d6', '#fbe3d4', '#fce2d2', '#fce0d0', '#fcdfcf',
        '#fcdecd', '#fdddcb', '#fddcc9', '#fddbc7', '#fdd9c4', '#fcd7c2', '#fcd5bf', '#fcd3bc',
        '#fbd0b9', '#fbceb7', '#fbccb4', '#facab1', '#fac8af', '#f9c6ac', '#f9c4a9', '#f9c2a7',
        '#f8bfa4', '#f8bda1', '#f8bb9e', '#f7b99c', '#f7b799', '#f7b596', '#f6b394', '#f6b191',
        '#f6af8e', '#f5ac8b', '#f5aa89', '#f5a886', '#f4a683', '#f3a481', '#f2a17f', '#f19e7d',
        '#f09c7b', '#ef9979', '#ee9677', '#ec9374', '#eb9172', '#ea8e70', '#e98b6e', '#e8896c',
        '#e6866a', '#e58368', '#e48066', '#e37e64', '#e27b62', '#e17860', '#df765e', '#de735c',
        '#dd7059', '#dc6e57', '#db6b55', '#da6853', '#d86551', '#d7634f', '#d6604d', '#d55d4c',
        '#d35a4a', '#d25849', '#d05548', '#cf5246', '#ce4f45', '#cc4c44', '#cb4942', '#c94741',
        '#c84440', '#c6413e', '#c53e3d', '#c43b3c', '#c2383a', '#c13639', '#bf3338', '#be3036',
        '#bd2d35', '#bb2a34', '#ba2832', '#b82531', '#b72230', '#b61f2e', '#b41c2d', '#b3192c',
        '#b1182b', '#ae172a', '#ab162a', '#a81529', '#a51429', '#a21328', '#9f1228', '#9c1127',
        '#991027', '#960f27', '#930e26', '#900d26', '#8d0c25', '#8a0b25', '#870a24', '#840924',
        '#810823', '#7f0823', '#7c0722', '#790622', '#760521', '#730421', '#700320', '#6d0220',
        '#6a011f', '#67001f'];

        if (cmap === "linearlab") {
            return linearLabHex;
        } else if (cmap === "inferno") {
            return infernoHex;
        } else if (cmap === "RdBu_r") {
            return RdBuHex;
        } else {
            return ["#000000", "#FFFFFF"];
        }
    }

    setColorStops(cmap){
        let colorHex = this.setColorMapHex(cmap);
        let stopLocations = colorHex.length;
        let colormap = new Array(stopLocations);
        for (let ii = 0; ii < stopLocations; ii++) {
            colormap[ii] = [ii / (stopLocations - 1), colorHex[ii]];
        }
        return colormap;
    }

    quantileClip(range){
        let quantLow, quantHigh, zQuantLow, zQuantHigh;
        [quantLow, quantHigh] = this.plotparams["percentile_clip"];
        [zQuantLow, zQuantHigh] = this.getMinMax(range);
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
    initMap() {
        // set the galaxy class self to a variable
        let _galthis = this.galthis;

        // get the ranges
        //let range  = this.getXRange();
        let xyrange, zrange;
        [xyrange, zrange]  = this.getRange();

        // get the min and max of the ranges
        let xymin, xymax, zmin, zmax;
        [xymin, xymax] = this.getMinMax(xyrange);
        [zmin, zmax] = this.getMinMax(zrange);

        // set null data and create new zrange, min, and max
        let data = this.setNull(this.data);
        zrange = data.map((o)=>{ return o[2]; });
        zrange = zrange.filter(this.filterRange);
        // [zmin, zmax] = this.getMinMax(zrange);
        [zmin, zmax] = this.quantileClip(zrange);

        let cmap = this.plotparams["cmap"];

        // make color bar symmetric
        if (this.plotparams["symmetric"]){
            let zabsmax = Math.max.apply(null, [Math.abs(zmin), Math.abs(zmax)]);
            [zmin, zmax] = [-zabsmax, zabsmax];
        }

        let cstops = this.setColorStops(cmap);

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
            credits: {enabled: false},
            title: {
                text: this.title.replace(/[_]/g, " "),
                style: {fontSize: "14px"}
            },
            navigation: {
                buttonOptions: {
                    theme: {fill: null}
                }
            },
            xAxis: {
                title: {text: 'Spaxel X'},
                minorGridLineWidth: 0,
                min: xymin,
                max: xymax,
                tickInterval: 1,
                tickLength: 0
            },
            yAxis:{
                title: {text: 'Spaxel Y'},
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
                labels: {align: 'center'},
                reversed: false,
                startOnTick: false,
                endOnTick: false,
                tickPixelInterval: 30,
                type: "linear",
            },
            plotOptions: {
                heatmap:{
                    nullColor: 'url(#custom-pattern)'  //'#A8A8A8'
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
                        strokeWidth: 0.3,
                        // fill: 'rgba(255, 255, 255, 1)'  // 'rgba(168, 168, 168, 0.3)'
                    }
                }]
            },
            legend: {
                align: 'right',
                layout: 'vertical',
                verticalAlign: 'middle',
                title: {text: this.parameter},
            },
            tooltip: {
                formatter: function () { return '<br>('+this.point.x+', '+this.point.y+'): <b>'+this.point.value+'</b><br>'; }
            },
            series:[{
                type: "heatmap",
                data: data,
                dataLabels: {enabled: false},
                events: {
                    click: (_galthis === null) ? undefined : function (event) { _galthis.getSpaxel(event); }
                }
            }]
        });
    }

}
