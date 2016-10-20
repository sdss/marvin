/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-10-20 22:51:56
*/

'use strict';

class Marvin {
    constructor(options) {
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
    setupRaven() {
        Raven.config('https://98bc7162624049ffa3d8d9911e373430@sentry.io/107924', {
            release: '0.2.0b1',
            // we highly recommend restricting exceptions to a domain in order to filter out clutter
            whitelistUrls: ['/(sas|api)\.sdss\.org/marvin/', '/(sas|api)\.sdss\.org/marvin2/'],
            includePaths: ['/https?:\/\/((sas|api)\.)?sdss\.org/marvin', '/https?:\/\/((sas|api)\.)?sdss\.org/marvin2']
        }).install();
    }
}

