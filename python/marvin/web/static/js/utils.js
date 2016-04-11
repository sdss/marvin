/*
* @Author: Brian Cherinka
* @Date:   2016-04-11 14:19:38
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-11 15:24:12
*/

'use strict';

// Javascript code for general things

var Utils,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Utils = (function() {

    marvin.Utils = Utils;

    // Constructor
    function Utils() {

        // in case constructor called without new
        if (false === (this instanceof Utils)) {
            return new Utils();
        }

        this.init();

        // event handlers

    }

    // initialize the object
    Utils.prototype.init = function init() {

    };

    // Build form
    Utils.prototype.buildForm = function buildForm() {
        var _len=arguments.length;
        var args = new Array(_len); for(var $_i = 0; $_i < _len; ++$_i) {args[$_i] = arguments[$_i];}
        var names = args[0];
        var form = {};
        $.each(args.slice(1),function(index,value) {
            form[names[index]] = value;
        });
        return form;
    }

    // Return unique elements of an Array
    Utils.prototype.unique = function unique(data) {
        var result = [];
        $.each(data, function(i, value) {
            if ($.inArray(value, result) == -1) result.push(value);
        });
        return result;
    };

    return Utils;
})();
