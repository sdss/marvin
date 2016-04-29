/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-28 23:33:27
*/

// Javascript code for general things

'use strict';

class Utils {

    // Constructor
    constructor() {
    }

    // Print
    print() {
        console.log('I am Utils!');
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

    // Scroll to div
    scrollTo(location) {
        if (location !== undefined) {
            var scrolldiv = $(location);
            $('html,body').animate({scrollTop:scrolldiv.offset().top},1500, 'easeInOutExpo');
        } else {
            $('html,body').animate({scrollTop:0},1500, 'easeInOutExpo');
        }

    }
}

