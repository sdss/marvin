/*
* @Author: Brian Cherinka
* @Date:   2016-04-25 13:56:19
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-09-28 13:25:11
*/

//jshint esversion: 6
'use strict';

class Table {

    // Constructor
    constructor(tablediv) {
        this.setTable(tablediv);

        // Event Handlers
        this.table.on('load-success.bs.table', this, this.setSuccessMsg);
        this.table.on('load-error.bs.table', this, this.setErrMsg);
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
            this.errdiv = this.table.siblings('#errdiv');
            this.tableerr = this.errdiv.find('#tableerror');
            this.tableerr.hide();
        }
    }

    // initialize a table
    initTable(url, data) {
        this.url = url;
        let cols;

        // if data
        if (data.columns !== null) {
            cols = this.makeColumns(data.columns);
        }

        console.log(data);
        console.log('cols', cols);
        // init the Bootstrap table
        this.table.bootstrapTable({
            classes: 'table table-bordered table-condensed table-hover',
            toggle : 'table',
            toolbar: '#toolbar',
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
            showColumns : true,
            showToggle : true,
            sortName: 'mangaid',
            sortOrder: 'asc',
            formatNoMatches: ()=>{ return "This table is empty..."; }
        });
    }

    // update the error div with a message
    updateMsg(msg) {
        let errmsg = `<strong>${msg}</strong>`;
        this.tableerr.html(errmsg);
        this.tableerr.show();
    }

    // set a table error message
    setErrMsg(event, status, res) {
        let _this = event.data;
        let extra = '';
        if (status === 502) {
            extra = 'bad server response retrieving web table.  likely uncaught error on server side.  check logs.';
        }
        let msg = `Status ${status} - ${res.statusText}: ${extra}`;
        _this.updateMsg(msg);
    }

    // set a table error message
    setSuccessMsg(event, data) {
        let _this = event.data;
        _this.tableerr.hide();
        if (data.status === -1) {
            _this.updateMsg(data.errmsg);
        }
    }

    // make the Table Columns
    makeColumns(columns) {
        let cols = [];
        columns.forEach((name, index)=>{
            let colmap = {};
            colmap.field = name;
            colmap.title = name;
            colmap.sortable = true;
            if (name.match('plateifu|mangaid')) {
                colmap.formatter = this.linkformatter;
            }
            cols.push(colmap);
        });
        return cols;
    }

    // Link Formatter
    linkformatter(value, row, index) {
        let url = Flask.url_for('galaxy_page.Galaxy:get', {'galid': value});
        let link = `<a href=${url} target='_blank'>${value}</a>`;
        return link;
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
