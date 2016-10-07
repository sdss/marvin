/*
* @Author: Brian Cherinka
* @Date:   2016-05-13 13:26:21
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-14 10:29:12
*/

'use strict';

class Search {

    // Constructor
    constructor() {
        this.searchform = $('#searchform');
        this.typeahead = $('#searchform .typeahead');
        this.returnparams = $('#returnparams');
        this.parambox = $('#parambox');
        this.searchbox = $("#searchbox");
    }

    // Print
    print() {
        console.log('I am Search!');
    }

    // Extract
    extractor(input) {
        var regexp = new RegExp('([^,]+)$');
        // parse input for newly typed text
        var result = regexp.exec(input);
        // select last entry after comma
        if(result && result[1]) {
            return result[1].trim();
        }
        return '';
    }

    // Initialize Query Param Typeahead
    initTypeahead(typediv, formdiv, url, fxn) {

        var _this = this;
        var typediv = (typediv === undefined) ? this.typeahead : $(typediv);
        var formdiv = (formdiv === undefined) ? this.searchform : $(formdiv);
        var typeurl = (url === undefined) ? Flask.url_for('search_page.getparams') : url;
        var afterfxn = (fxn === undefined) ? null : fxn;

        function customQueryTokenizer(str) {
            var newstr = str.toString();
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
            filter: function(qpars) {
                return qpars;
            }
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
            var currenttext = this.$element.val();
            var removedtemptype = currenttext.replace(/[^,]*$/,'');
            var newtext = removedtemptype+item+', ';
            return newtext;
        },
        matcher: function (item) {
            // used to determined if a query matches an item
            var tquery = _this.extractor(this.query);
            if(!tquery) return false;
            return ~item.toLowerCase().indexOf(tquery.toLowerCase())
        },
        highlighter: function (item) {
          // used to highlight autocomplete results ; returns html
          var oquery = _this.extractor(this.query);
          var query = oquery.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
          return item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
            return '<strong>' + match + '</strong>'
          })
        }
        });
    }
}
