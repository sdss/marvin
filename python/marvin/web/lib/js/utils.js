/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-14 11:43:57
*/

// Javascript code for general things

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Utils = function () {

    // Constructor

    function Utils() {
        _classCallCheck(this, Utils);
    }

    // Print


    _createClass(Utils, [{
        key: 'print',
        value: function print() {
            console.log('I am Utils!');
        }

        // Build a Form

    }, {
        key: 'buildForm',
        value: function buildForm(keys) {
            var args = Array.prototype.slice.call(arguments, 1);
            var form = {};
            keys.forEach(function (key, index) {
                form[key] = args[index];
            });
            return form;
        }

        // Unique values

    }, {
        key: 'unique',
        value: function unique(data) {
            return new Set(data);
        }
    }]);

    return Utils;
}();
