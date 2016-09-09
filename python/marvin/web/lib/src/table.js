/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-09 16:52:45
*/

'use strict';

class Table {

    // Constructor
    constructor(tablediv) {
        this.setTable(tablediv);
    }

    // Print
    print() {
        console.log('I am Table!');
    }

    // Set the initial Table
    setTable(tablediv) {
        if (tablediv !== undefined) {
            console.log('setting the table');
            this.table = tablediv;
        }
    }

    // initialize a table
    initTable(url, data) {
        this.url = url;

        // if data
        if (data.columns !== null) {
            var cols = this.makeColumns(data.columns);
        }

        // init the Bootstrap table
        this.table.bootstrapTable({
            classes: 'table table-bordered table-condensed table-hover',
            toggle : 'table',
            pagination : true,
            pageSize: 10,
            pageList : '[10, 20, 50]',
            sidePagination: 'server',
            method: 'post',
            contentType: "application/x-www-form-urlencoded",
            data: data.rows,
            totalRows: data.total,
            columns: cols,
            url: url,
            search : true,
            showColumns : true,
            showToggle : true,
            sortName: 'cube.mangaid',
            sortOrder: 'asc',
            formatNoMatches: function () {
                return "This table is empty...";
            }
        })
    }

    // make the Table Columns
    makeColumns(columns) {
        var cols = [];
        columns.forEach(function (name, index) {
            var colmap = {};
            colmap['field'] = name;
            colmap['title'] = name;
            colmap['sortable'] = true;
            cols.push(colmap);
        });
        return cols;
    }

    // Handle the Bootstrap table JSON response
    handleResponse(results) {
        // load the bootstrap table div
        //console.log(this.table, this.table===null, this);
        if (this.table === null) {
            this.setTable();
        }
        this.table = $('#table');
        //console.log('after', this.table, this.table===null, $('#table'));
        // Get new columns
        var cols = results.columns;
        var cols = [];
        results.columns.forEach(function (name, index) {
            var colmap = {};
            colmap['field'] = name;
            colmap['title'] = name;
            colmap['sortable'] = true;
            cols.push(colmap);
        });

        // Load new options
        this.table.bootstrapTable('refreshOptions', {'columns': cols, 'totalRows': results.total});

        return results;
    }

}
