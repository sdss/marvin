/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-06-04 02:03:56
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Table = function () {

    // Constructor
    function Table(tablediv) {
        _classCallCheck(this, Table);

        this.setTable(tablediv);
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
        value: function setTable(tablediv) {
            if (tablediv !== undefined) {
                console.log('setting the table');
                this.table = tablediv;
            }
        }

        // initialize a table

    }, {
        key: 'initTable',
        value: function initTable(url, data) {
            this.url = url;
            var cols = void 0;

            // if data
            if (data.columns !== null) {
                cols = this.makeColumns(data.columns);
            }

            // init the Bootstrap table
            this.table.bootstrapTable({
                classes: 'table table-bordered table-condensed table-hover',
                toggle: 'table',
                toolbar: '#toolbar',
                pagination: true,
                pageSize: 10,
                pageList: '[10, 20, 50]',
                sidePagination: 'server',
                method: 'post',
                contentType: "application/x-www-form-urlencoded",
                data: data.rows,
                totalRows: data.total,
                columns: cols,
                url: url,
                showColumns: true,
                showToggle: true,
                sortName: 'cube.mangaid',
                sortOrder: 'asc',
                formatNoMatches: function formatNoMatches() {
                    return "This table is empty...";
                }
            });
        }

        // make the Table Columns

    }, {
        key: 'makeColumns',
        value: function makeColumns(columns) {
            var _this = this;

            var cols = [];
            columns.forEach(function (name, index) {
                var colmap = {};
                colmap.field = name;
                colmap.title = name;
                colmap.sortable = true;
                if (name.match('cube.plateifu|cube.mangaid')) {
                    colmap.formatter = _this.linkformatter;
                }
                cols.push(colmap);
            });
            return cols;
        }

        // Link Formatter

    }, {
        key: 'linkformatter',
        value: function linkformatter(value, row, index) {
            var url = Flask.url_for('galaxy_page.Galaxy:get', { 'galid': value });
            var link = '<a href=' + url + ' target=\'_blank\'>' + value + '</a>';
            return link;
        }

        // Handle the Bootstrap table JSON response

    }, {
        key: 'handleResponse',
        value: function handleResponse(results) {
            // load the bootstrap table div
            if (this.table === null) {
                this.setTable();
            }
            this.table = $('#table');
            // Get new columns
            var cols = results.columns;
            cols = [];
            results.columns.forEach(function (name, index) {
                var colmap = {};
                colmap.field = name;
                colmap.title = name;
                colmap.sortable = true;
                cols.push(colmap);
            });

            // Load new options
            this.table.bootstrapTable('refreshOptions', { 'columns': cols, 'totalRows': results.total });

            return results;
        }
    }]);

    return Table;
}();
