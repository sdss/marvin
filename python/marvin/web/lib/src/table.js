/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-26 15:59:14
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
        if (data) {

        }

        // init the Bootstrap table
        this.table.bootstrapTable({
            classes: 'table table-bordered table-condensed table-hover',
            toggle : 'table',
            pagination : true,
            pageList : '[5, 10, 20]',
            sidePagination: 'server',
            method: 'post',
            contentType: "application/json",
            columns: cols,
            url: url,
            search : true,
            showColumns : true,
            showToggle : true,
            sortName: 'mangaid',
            sortOrder: 'asc',
            formatNoMatches: function () {
                return "This table is empty...";
            }
        })
    }

    // make the Table Columns
    makeColumns(columns) {

    }

    // Handle the Bootstrap table JSON response
    handleResponse(results) {
        console.log('table results', results);
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
        console.log(cols);

        // Load new options
        this.table.bootstrapTable('refreshOptions', {'columns': cols});

        return results;
    }

}
