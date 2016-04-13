/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-13 18:07:00
*/

// Javascript code for general things

'use strict';

class Utils {

    // Constructor
    constructor() {
    }

    // Build a Form
    buildForm(keys) {
        var args = Array.prototype.slice.call(arguments, 1);
        var form = {};
        keys.forEach(function (key, index) {
            form[key] = args[index];
        });
        return form;
    }

    // Unique values
    unique(data) {
        return new Set(data);
    }
}
