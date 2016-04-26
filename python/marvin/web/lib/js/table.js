/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-26 00:26:56
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Table = function () {

    // Constructor

    function Table() {
        _classCallCheck(this, Table);

        this.table = null;
    }

    // Print


    _createClass(Table, [{
        key: 'print',
        value: function print() {
            console.log('I am Table!');
        }

        // Set the initial Table

    }, {
        key: 'setTable',
        value: function setTable() {
            console.log('setting the table');
            this.table = $('#table');
        }

        // make the Table Columns

    }, {
        key: 'makeColumns',
        value: function makeColumns(columns) {}

        // Handle the Bootstrap table JSON response

    }, {
        key: 'handleResponse',
        value: function handleResponse(results) {
            console.log('table results', results);
            // load the bootstrap table div
            console.log(this.table, this.table === null, this);
            if (this.table === null) {
                this.setTable();
            }
            this.table = $('#table');
            console.log('after', this.table, this.table === null, $('#table'));
            // Get new columns
            var cols = results.columns;
            console.log(cols);

            // Load new options
            //this.table.bootstrapTable('refreshOptions', {'columns': cols});

            return results;
        }
    }]);

    return Table;
}();
