/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian
* @Last Modified time: 2016-05-08 19:35:12
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
        //this.utils.initPopOvers();

        // load the header
        this.header = new Header();
        this.header.print();
    }
}

