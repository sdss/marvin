/*
* @Author: Brian Cherinka
* @Date:   2016-05-13 13:26:21
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-06-28 11:59:04
*/

//jshint esversion: 6
'use strict';

class Search {

    // Constructor
    constructor() {
        this.searchform = $('#searchform');
        this.typeahead = $('#searchform .typeahead');
        this.returnparams = $('#returnparams');
        this.parambox = $('#parambox');
        this.searchbox = $("#searchbox");

        this.builder = $('#builder');
        this.sqlalert = $('#sqlalert');
        this.getsql = $('#get-sql');
        this.resetsql = $('#reset-sql');
        this.runsql = $('#run-sql');

        // Event Handlers
        this.getsql.on('click', this, this.getSQL);
        this.resetsql.on('click', this, this.resetSQL);
        this.runsql.on('click', this, this.runSQL);

    }

    // Print
    print() {
        console.log('I am Search!');
    }

    // Extract
    extractor(input) {
        let regexp = new RegExp('([^,]+)$');
        // parse input for newly typed text
        let result = regexp.exec(input);
        // select last entry after comma
        if(result && result[1]) {
            return result[1].trim();
        }
        return '';
    }

    // Initialize Query Param Typeahead
    initTypeahead(typediv, formdiv, url, fxn) {

        const _this = this;
        let typeurl;
        typediv = (typediv === undefined) ? this.typeahead : $(typediv);
        formdiv = (formdiv === undefined) ? this.searchform : $(formdiv);
        // get the typeahead search page getparams url
        try {
            typeurl = (url === undefined) ? Flask.url_for('search_page.getparams', {'paramdisplay':'best'}) : url;
        } catch (error) {
            Raven.captureException(error);
            console.error('Error getting search getparams url:',error);
        }
        const afterfxn = (fxn === undefined) ? null : fxn;

        function customQueryTokenizer(str) {
            let newstr = str.toString();
            return [_this.extractor(newstr)];
        };

        // create the bloodhound engine
        this.queryparams = new Bloodhound({
        datumTokenizer: Bloodhound.tokenizers.whitespace,
        //queryTokenizer: Bloodhound.tokenizers.whitespace,
        queryTokenizer: customQueryTokenizer,
        prefetch: typeurl,
        remote: {
            url: typeurl,
            filter: (qpars)=>{ return qpars; }
        }
        });

        // initialize the bloodhound suggestion engine
        this.queryparams.initialize();

        // init the search typeahead
        typediv.typeahead('destroy');
        typediv.typeahead(
        {
        showHintOnFocus: true,
        items: 'all',
        source:this.queryparams.ttAdapter(),
        updater: function(item) {
            // used to updated the input box with selected option
            // item = selected item from dropdown
            let currenttext = this.$element.val();
            let removedtemptype = currenttext.replace(/[^,]*$/,'');
            let newtext = removedtemptype+item+', ';
            return newtext;
        },
        matcher: function (item) {
            // used to determined if a query matches an item
            let tquery = _this.extractor(this.query);
            console.log('query', this.query);
            console.log(tquery);
            if(!tquery) return false;
            return ~item.toLowerCase().indexOf(tquery.toLowerCase());
        },
        highlighter: function (item) {
          // used to highlight autocomplete results ; returns html
          let oquery = _this.extractor(this.query);
          let query = oquery.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
          return item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
            return '<strong>' + match + '</strong>';
          });
        }
        });
    }

    // Setup Query Builder
    setupQB(params) {
        $('.modal-dialog').draggable(); // makes the modal dialog draggable

        // set some parameters
        this.builder.params = params;
        this.spaxelprops = this.builder.params.map(i => (i.id.includes('spaxelprop')) ? i.id.split('.')[1] : null).filter(i => ![null,'x','y'].includes(i));

        // init the query builder
        this.builder.queryBuilder({plugins:['bt-selectpicker', 'not-group', 'invert'], filters:params,
            operators:['equal', 'not_equal', 'less', 'less_or_equal', 'greater', 'greater_or_equal',
                       'between', 'contains', 'begins_with', 'ends_with']});
    }

    // load the Query Builder with SQL
    loadQB(sql) {

        // patch incoming sql with spaxelprop parameter name
        sql = sql.replace(/cleanspaxelprop\d{1,2}/i, 'spaxelprop');
        this.spaxelprops.filter(i => (sql.includes(i) & !sql.includes('spaxelprop.'+i)) ? sql = sql.replace(i, 'spaxelprop.'+i): '');

        let valid = this.builder.queryBuilder('validate', sql);
        try {
            this.builder.queryBuilder('setRulesFromSQL', sql);
        } catch (error) {
            let msg = 'Error reloading sql filter into the Query Builder';
            if (!valid) {
                msg = 'Error. Invalid SQL: Cannot reload. ' + error.message;    
            }
            this.sqlalert.html("<p class='text-center text-danger'>" + msg + "</p>");
        }
    }

    // Get the SQL from the QB
    getSQL(event) {
        let _this = event.data;
        try {
          var result = _this.builder.queryBuilder('getSQL', false);
            if (result.sql.length) {
              _this.sqlalert.html("");
              // remove the quotations
              let newsql = result.sql.replace(/[']+/g, "");
              // replace any like and percents with = and *
              let likeidx = newsql.indexOf('LIKE');
              if (likeidx !== -1) {
                newsql = newsql.replace('LIKE(', '= ').replace(/[%]/g, '*');
                let idx = newsql.indexOf(')', likeidx);
                newsql = newsql.replace(newsql.charAt(idx), " ");
              }
              _this.searchbox.val(newsql);
            }
        } catch (error) {
          _this.sqlalert.html("<p class='text-center text-danger'>Must provide valid input.</p>");
        }
    }

    // Reset the SQL in SearchBox and the QB
    resetSQL(event) {
        let _this = event.data;
       _this.searchbox.val("");
       _this.sqlalert.html("");
       _this.builder.queryBuilder('reset');
    }

    // Run the Query from the QB
    runSQL(event) {
        let _this = event.data;
        if (_this.searchbox.val() === "") {
            _this.sqlalert.html("<p class='text-center text-danger'>You must generate SQL first!</p>");
        } else {
            _this.searchform.submit();
        }
    }
}
