/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-09-28 13:25:11
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

        // Event Handlers
        this.table.on('load-success.bs.table', this, this.setSuccessMsg);
        this.table.on('load-error.bs.table', this, this.setErrMsg);
        $('#export-json').on('click', this, this.exportTable);
        $('#export-csv').on('click', this, this.exportTable);
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
                this.errdiv = this.table.siblings('#errdiv');
                this.tableerr = this.errdiv.find('#tableerror');
                this.tableerr.hide();
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

            this.data = data;
            this.initPageSize = 10;
            console.log(data);
            console.log('cols', cols);

            // init the Bootstrap table
            this.table.bootstrapTable({
                classes: 'table table-bordered table-condensed table-hover',
                toggle: 'table',
                toolbar: '#toolbar',
                pagination: true,
                pageSize: this.initPageSize,
                pageList: '[10, 20, 50]',
                sidePagination: 'server',
                method: 'post',
                contentType: "application/x-www-form-urlencoded",
                data: data.rows.slice(0, this.initPageSize),
                totalRows: data.total,
                columns: cols,
                deferUrl: url,
                showColumns: true,
                showToggle: true,
                queryParams: this.queryParams,
                sortName: 'mangaid',
                sortOrder: 'asc',
                usePipeline: true,
                formatNoMatches: function formatNoMatches() {
                    return "This table is empty...";
                }
            });
        }

        // send additional query parameters with each request

    }, {
        key: 'queryParams',
        value: function queryParams(params) {
            this.table = $('#searchtable');
            var options = this.table.bootstrapTable('getOptions');
            console.log('table opts', options);
            console.log('parmas', params);

            // this is necessary to ensure the backend has the validated API parameters it needs
            // the table pipeline extension renames some of the query parameters
            if (options.usePipeline) {
                params.order = params.sortOrder;
                params.sort = params.sortName;
            }
            return params;
        }

        // update the error div with a message

    }, {
        key: 'updateMsg',
        value: function updateMsg(msg) {
            var errmsg = '<strong>' + msg + '</strong>';
            this.tableerr.html(errmsg);
            this.tableerr.show();
        }

        // set a table error message

    }, {
        key: 'setErrMsg',
        value: function setErrMsg(event, status, res) {
            var _this = event.data;
            var extra = '';
            if (status === 502) {
                extra = 'bad server response retrieving web table.  likely uncaught error on server side.  check logs.';
            }
            var msg = 'Status ' + status + ' - ' + res.statusText + ': ' + extra;
            _this.updateMsg(msg);
        }

        // set a table error message

    }, {
        key: 'setSuccessMsg',
        value: function setSuccessMsg(event, data) {
            var _this = event.data;
            _this.tableerr.hide();
            if (data.status === -1) {
                _this.updateMsg(data.errmsg);
            }
        }

        // make the Table Columns

    }, {
        key: 'makeColumns',
        value: function makeColumns(columns) {
            var _this2 = this;

            var cols = [];
            columns.forEach(function (name, index) {
                var colmap = {};
                colmap.field = name;
                colmap.title = name;
                colmap.sortable = true;
                if (name.match('plateifu|mangaid')) {
                    colmap.formatter = _this2.linkformatter;
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
            this.table = $('#searchtable');
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

        // export the table data

    }, {
        key: 'exportTable',
        value: function exportTable(event) {
            var _this = event.data;
            var filetype = event.currentTarget.name;

            var options = $('#searchtable').bootstrapTable('getOptions');
            var url = options.deferUrl;

            var params = { limit: options.totalRows, export: true, filetype: filetype,
                offset: 0, sort: 'mangaid', order: 'asc' };

            var form = { 'body': JSON.stringify(params),
                'method': 'POST', 'headers': { 'Content-Type': 'application/json' } };

            // toggle on export displays
            $('#export-btn').children('button').addClass('btn-warning');
            $('#export-load').show();

            // fetch the full query results
            return fetch(url, form).then(function (response) {
                return response.blob();
            }).then(function (blob) {
                // convert blob into a file link and prompt to download
                var url = window.URL.createObjectURL(blob);
                var link = document.createElement('a');
                document.body.appendChild(link);
                link.style = "display: none";
                link.href = url;
                link.download = "marvin_table." + filetype;
                link.click();
            }).then(function () {
                // toggle off export displays
                $('#export-btn').children('button').removeClass('btn-warning');
                $('#export-load').hide();
            });
        }
    }]);

    return Table;
}();
