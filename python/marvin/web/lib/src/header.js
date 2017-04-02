/*
* @Author: Brian Cherinka
* @Date:   2016-04-26 21:47:05
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 13:09:41
*/

//jshint esversion: 6
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

        const _this = this;
        typediv = (typediv === undefined) ? this.typeahead : $(typediv);
        formdiv = (formdiv === undefined) ? this.galidform : $(formdiv);
        let typeurl = (url === undefined) ? Flask.url_for('index_page.getgalidlist') : url;
        let afterfxn = (fxn === undefined) ? null : fxn;

        // create the bloodhound engine
        this.galids = new Bloodhound({
        datumTokenizer: Bloodhound.tokenizers.whitespace,
        queryTokenizer: Bloodhound.tokenizers.whitespace,
        //local:  ["(A)labama","Alaska","Arizona","Arkansas","Arkansas2","Barkansas", 'hello'],
        prefetch: typeurl,
        remote: {
            url: typeurl,
            filter: (galids)=>{ return galids; }
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
        afterSelect: ()=>{ formdiv.submit(); }
        });
    }

    // Select the MPL version on the web
    selectMPL(event) {
        const _this = event.data;
        let url = 'index_page.selectmpl';
        let verform = m.utils.serializeForm('#mplform');
        _this.sendAjax(verform, url, _this.reloadPage);
    }

    // Reload the Current Page
    reloadPage() {
        location.reload(true);
    }

    // Send an AJAX request
    sendAjax(form, url, fxn) {
        const _this = this;
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

