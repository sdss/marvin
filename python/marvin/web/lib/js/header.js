/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-27 23:23:03
*/

'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Header = function () {

    // Constructor

    function Header() {
        _classCallCheck(this, Header);

        this.navbar = $('.navbar');
        this.galidform = $('#headform');
        this.typeahead = $('.galids .typeahead');

        this.initTypeahead();
    }

    // Print


    _createClass(Header, [{
        key: 'print',
        value: function print() {
            console.log('I am Header!', this.galids, this.typeahead);
        }

        // Initialize the Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead() {

            var _this = this;

            // create the bloodhound engine
            this.galids = new Bloodhound({
                datumTokenizer: Bloodhound.tokenizers.whitespace,
                queryTokenizer: Bloodhound.tokenizers.whitespace,
                //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
                prefetch: Flask.url_for('index_page.getgalidlist'),
                remote: {
                    url: Flask.url_for('index_page.getgalidlist'),
                    filter: function filter(galids) {
                        return galids;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.galids.initialize();

            $('.typeahead').typeahead('destroy');
            $('.typeahead').typeahead({
                showHintOnFocus: true,
                source: this.galids.ttAdapter(),
                afterSelect: function afterSelect() {
                    _this.galidform.submit();
                }
            });
        }
    }]);

    return Header;
}();
