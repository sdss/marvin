/*
* @Author: Brian Cherinka
* @Date:   2016-05-13 13:26:21
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-09-14 10:29:12
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Search = function () {

    // Constructor
    function Search() {
        _classCallCheck(this, Search);

        this.searchform = $('#searchform');
        this.typeahead = $('#searchform .typeahead');
        this.returnparams = $('#returnparams');
        this.parambox = $('#parambox');
        this.searchbox = $("#searchbox");
    }

    // Print


    _createClass(Search, [{
        key: 'print',
        value: function print() {
            console.log('I am Search!');
        }

        // Extract

    }, {
        key: 'extractor',
        value: function extractor(input) {
            var regexp = new RegExp('([^,]+)$');
            // parse input for newly typed text
            var result = regexp.exec(input);
            // select last entry after comma
            if (result && result[1]) {
                return result[1].trim();
            }
            return '';
        }

        // Initialize Query Param Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead(typediv, formdiv, url, fxn) {

            var _this = this;
            var typediv = typediv === undefined ? this.typeahead : $(typediv);
            var formdiv = formdiv === undefined ? this.searchform : $(formdiv);
            var typeurl = url === undefined ? Flask.url_for('search_page.getparams') : url;
            var afterfxn = fxn === undefined ? null : fxn;

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
                    filter: function filter(qpars) {
                        return qpars;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.queryparams.initialize();

            // init the search typeahead
            typediv.typeahead('destroy');
            typediv.typeahead({
                showHintOnFocus: true,
                items: 'all',
                source: this.queryparams.ttAdapter(),
                updater: function updater(item) {
                    // used to updated the input box with selected option
                    // item = selected item from dropdown
                    var currenttext = this.$element.val();
                    var removedtemptype = currenttext.replace(/[^,]*$/, '');
                    var newtext = removedtemptype + item + ', ';
                    return newtext;
                },
                matcher: function matcher(item) {
                    // used to determined if a query matches an item
                    var tquery = _this.extractor(this.query);
                    if (!tquery) return false;
                    return ~item.toLowerCase().indexOf(tquery.toLowerCase());
                },
                highlighter: function highlighter(item) {
                    // used to highlight autocomplete results ; returns html
                    var oquery = _this.extractor(this.query);
                    var query = oquery.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
                    return item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
                        return '<strong>' + match + '</strong>';
                    });
                }
            });
        }
    }]);

    return Search;
}();
