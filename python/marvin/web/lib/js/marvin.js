/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-28 10:24:41
*/

//jshint esversion: 6
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

        // check the browser on page load
        this.window = $(window);
        this.window.on('load', this.checkBrowser);
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

        // check the browser for banner display

    }, {
        key: 'checkBrowser',
        value: function checkBrowser(event) {
            var _this = event.data;
            if (!!navigator.userAgent.match(/Version\/[\d\.]+.*Safari/)) {
                m.utils.marvinBanner('We have detected that you are using Safari. Some features may not work as expected. We recommend using Chrome or Firefox.', 1, 'safari_banner', 'http://sdss-marvin.readthedocs.io/en/latest/known-issues.html#known-browser');
            }
        }
    }]);

    return Marvin;
}();
