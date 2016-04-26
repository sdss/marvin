/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-25 13:58:27
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

        // set up table functions
        this.table = new Table();
        this.table.print();
    }
}

