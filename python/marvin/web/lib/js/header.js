/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 13:09:41
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Header = function () {

    // Constructor
    function Header() {
        _classCallCheck(this, Header);

        this.navbar = $('.navbar');
        this.galidform = $('#headform');
        this.typeahead = $('#headform .typeahead');
        this.mplform = $('#mplform');
        this.mplselect = $('#mplselect');

        this.initTypeahead();

        //Event Handlers
        this.mplselect.on('change', this, this.selectMPL);
    }

    // Print


    _createClass(Header, [{
        key: 'print',
        value: function print() {
            console.log('I am Header!');
        }

        // Initialize the Typeahead

    }, {
        key: 'initTypeahead',
        value: function initTypeahead(typediv, formdiv, url, fxn) {

            var _this = this;
            typediv = typediv === undefined ? this.typeahead : $(typediv);
            formdiv = formdiv === undefined ? this.galidform : $(formdiv);
            var typeurl = url === undefined ? Flask.url_for('index_page.getgalidlist') : url;
            var afterfxn = fxn === undefined ? null : fxn;

            // create the bloodhound engine
            this.galids = new Bloodhound({
                datumTokenizer: Bloodhound.tokenizers.whitespace,
                queryTokenizer: Bloodhound.tokenizers.whitespace,
                //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
                prefetch: typeurl,
                remote: {
                    url: typeurl,
                    filter: function filter(galids) {
                        return galids;
                    }
                }
            });

            // initialize the bloodhound suggestion engine
            this.galids.initialize();

            typediv.typeahead('destroy');
            typediv.typeahead({
                showHintOnFocus: true,
                items: 30,
                source: this.galids.ttAdapter(),
                afterSelect: function afterSelect() {
                    formdiv.submit();
                }
            });
        }

        // Select the MPL version on the web

    }, {
        key: 'selectMPL',
        value: function selectMPL(event) {
            var _this = event.data;
            var url = 'index_page.selectmpl';
            var verform = m.utils.serializeForm('#mplform');
            _this.sendAjax(verform, url, _this.reloadPage);
        }

        // Reload the Current Page

    }, {
        key: 'reloadPage',
        value: function reloadPage() {
            location.reload(true);
        }

        // Send an AJAX request

    }, {
        key: 'sendAjax',
        value: function sendAjax(form, url, fxn) {
            var _this = this;
            $.post(Flask.url_for(url), form, 'json').done(function (data) {
                // reload the current page, this re-instantiates a new Header with new version info from session
                if (data.result.status == 1) {
                    fxn();
                    _this.galids.clearPrefetchCache();
                    _this.galids.initialize();
                } else {
                    alert('Failed to set the versions! ' + data.result.msg);
                }
            }).fail(function (data) {
                alert('Failed to set the versions! Problem with Flask setversion. ' + data.result.msg);
            });
        }
    }]);

    return Header;
}();
