/*
* @Author: Brian Cherinka
* @Date:   2016-04-13 11:24:07
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-26 13:44:41
*/

'use strict';

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Marvin = function Marvin(options) {
    _classCallCheck(this, Marvin);

    // set options
    //_.defaults(options, {fruit: "strawberry"})
    this.options = options;

    // set up utility functions
    this.utils = new Utils();
    this.utils.print();
};
