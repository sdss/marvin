/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:39:10
*/

//jshint esversion: 6
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
        let cols;

        // if data
        if (data.columns !== null) {
            let cols = this.makeColumns(data.columns);
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
            formatNoMatches: ()=>{ return "This table is empty..."; }
        });
    }

    // make the Table Columns
    makeColumns(columns) {
        let cols = [];
        columns.forEach((name, index)=>{
            let colmap = {};
            colmap.field = name;
            colmap.title = name;
            colmap.sortable = true;
            cols.push(colmap);
        });
        return cols;
    }

    // Handle the Bootstrap table JSON response
    handleResponse(results) {
        // load the bootstrap table div
        if (this.table === null) {
            this.setTable();
        }
        this.table = $('#table');
        // Get new columns
        let cols = results.columns;
        cols = [];
        results.columns.forEach((name, index)=>{
            let colmap = {};
            colmap.field = name;
            colmap.title = name;
            colmap.sortable = true;
            cols.push(colmap);
        });

        // Load new options
        this.table.bootstrapTable('refreshOptions', {'columns': cols, 'totalRows': results.total});

        return results;
    }

}
