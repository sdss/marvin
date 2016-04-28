/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-27 23:23:03
*/

'use strict';

class Header {

    // Constructor
    constructor() {
        this.navbar = $('.navbar');
        this.galidform = $('#headform');
        this.typeahead = $('.galids .typeahead');

        this.initTypeahead();
    }

    // Print
    print() {
        console.log('I am Header!', this.galids, this.typeahead);
    }

    // Initialize the Typeahead
    initTypeahead() {

        var _this = this;

        // create the bloodhound engine
        this.galids = new Bloodhound({
        datumTokenizer: Bloodhound.tokenizers.whitespace,
        queryTokenizer: Bloodhound.tokenizers.whitespace,
        //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
        prefetch: Flask.url_for('index_page.getgalidlist'),
        remote: {
            url: Flask.url_for('index_page.getgalidlist'),
            filter: function(galids) {
                return galids;
            }
        }
        });

        // initialize the bloodhound suggestion engine
        this.galids.initialize();

        $('.typeahead').typeahead('destroy')
        $('.typeahead').typeahead(
        {
        showHintOnFocus: true,
        source:this.galids.ttAdapter(),
        afterSelect: function() {
            _this.galidform.submit();
        }
        });
    }

}

