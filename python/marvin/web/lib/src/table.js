/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-25 14:23:57
*/

'use strict';

class Table {

    // Constructor
    constructor() {
        this.table = null;
    }

    // Print
    print() {
        console.log('I am Table!');
    }

    // Set the initial Table
    setTable() {
        console.log('setting the table');
        this.table = $('#table');
    }

    // make the Table Columns
    makeColumns(columns) {

    }

    // Handle the Bootstrap table JSON response
    handleResponse(results) {
        console.log('table results', results);
        // load the bootstrap table div
        console.log(this.table, this.table===null);
        if (this.table === null) {
            this.setTable();
        }
        this.table = $('#table');
        console.log('after', this.table, this.table===null, $('#table'));
        // Get new columns
        var cols = results.columns;
        console.log(cols);

        // Load new options
        this.table.bootstrapTable('refreshOptions', {'columns': cols});

        return results;
    }

}
