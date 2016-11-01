/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-10-12 16:44:53
*/

'use strict';

class Header {

    // Constructor
    constructor() {
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
    print() {
        console.log('I am Header!');
    }

    // Initialize the Typeahead
    initTypeahead(typediv, formdiv, url, fxn) {

        var _this = this;
        var typediv = (typediv === undefined) ? this.typeahead : $(typediv);
        var formdiv = (formdiv === undefined) ? this.galidform : $(formdiv);
        var typeurl = (url === undefined) ? Flask.url_for('index_page.getgalidlist') : url;
        var afterfxn = (fxn === undefined) ? null : fxn;

        // create the bloodhound engine
        this.galids = new Bloodhound({
        datumTokenizer: Bloodhound.tokenizers.whitespace,
        queryTokenizer: Bloodhound.tokenizers.whitespace,
        //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
        prefetch: typeurl,
        remote: {
            url: typeurl,
            filter: function(galids) {
                return galids;
            }
        }
        });

        // initialize the bloodhound suggestion engine
        this.galids.initialize();

        typediv.typeahead('destroy');
        typediv.typeahead(
        {
        showHintOnFocus: true,
        items: 30,
        source:this.galids.ttAdapter(),
        afterSelect: function() {
            formdiv.submit();
        }
        });
    }

    // Select the MPL version on the web
    selectMPL(event) {
        var _this = event.data;
        var url = 'index_page.selectmpl';
        var verform = m.utils.serializeForm('#mplform');
        console.log('setting new mpl', verform);
        _this.sendAjax(verform, url, _this.reloadPage);
    }

    // Reload the Current Page
    reloadPage() {
        location.reload(true);
    }

    // Send an AJAX request
    sendAjax(form, url, fxn) {
        var _this = this;
        $.post(Flask.url_for(url), form, 'json')
        .done(function(data){
            // reload the current page, this re-instantiates a new Header with new version info from session
            if (data.result.status == 1) {
                fxn();
                _this.galids.clearPrefetchCache();
                _this.galids.initialize();
            } else {
                alert('Failed to set the versions! '+data.result.msg);
            }
        })
        .fail(function(data){
            alert('Failed to set the versions! Problem with Flask setversion. '+data.result.msg);
        });

    }

}

