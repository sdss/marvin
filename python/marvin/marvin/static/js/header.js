

var Header,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Header = (function () {

    function Header(vermode,options) {

        // in case constructor called without new
        if (false === (this instanceof Header)) {
            return new Header();
        }
        
        this.init(vermode,options);
        
        // Event Handlers
        $('#idselect').on('change',this,this.toggleSearchType);
        $('.verselecttype').on('change',this,this.sendVersionInfo);
        $('#marvinmodeselect').on('change',this,this.toggleMode);
    }
    
    // initialize the object
    Header.prototype.init = function(vermode,options) {
        this.versionmode = vermode;
        this.versionid = null;
        this.searchid = null;
        this.searchtext = null;
        this.searchoptions = options;
        this.typetext = null;
        this.innerhtml = null;
        this.marvinmode = null;

        this.setParams();
        this.showVersion();
        this.initTypeahead();
    };
    
    // test print
    Header.prototype.print = function() {
        console.log('We are now printing header info: vermode', this.versionmode, this.versionid, this.searchid);
    };

    // initialize the typeahead
    Header.prototype.initTypeahead = function() {
        console.log('init typeahead', $('.typeahead'), $('#idtext').attr('placeholder'));
        options = {
            source:["Blah","hello","test"]
        };
        $('.typeahead').typeahead(options);
    };

    // Set the header parameters
    Header.prototype.setParams = function() {
        this.versionid = (this.versionmode.search('MPL') > -1) ? 'mpl' : 'drpdap';
        this.searchid = $('#idselect option:selected').attr('id');
        this.searchtext = (this.searchid == 'plateid') ? 'Plate ID' : 'MaNGA ID';
        this.typetext = (this.versionid=='mpl') ? 'MPL' : 'DRP/DAP';
        this.resetVersionButtonHtml();
        this.resetSearchText();
    };

    // Show the version display
    Header.prototype.showVersion = function() {
        $('.vertype').hide();
        $('#by'+this.versionid).show();        
    };

    // Check which version div is visible
    Header.prototype.isVisible = function(id) {
        return $('#by'+id).is(':visible');
    }

    // Get the version div that is visible 
    Header.prototype.getVisibleVersionID = function() {
        return $('.vertype:visible').attr('id');
    }

    // Build version form
    Header.prototype.buildForm = function() {
        var verform = $('#verform').serializeArray();
        verform.push({'name':'vermode','value':this.versionid});
        return verform;
    };

    // Send the selected Version Info 
    Header.prototype.sendVersionInfo = function(event) {
        var _this = event.data;
        var verform = _this.buildForm()
        _this.sendAjax(verform,'/marvin/setversion/', _this.reloadPage);
    };

    // Reset the version button text
    Header.prototype.resetVersionButtonHtml = function() {
        var but = $('#verbut');
        this.innerhtml = 'Set Version By: '+ this.typetext +' <span class="caret"></span>';
        but.html(this.innerhtml);
    };

    // Reset the search input placeholder text
    Header.prototype.resetSearchText = function() {
        $('#idtext').attr('placeholder',this.searchtext);
        $('#idtext').attr('name',this.searchid);        
    }

    // Toggle the Version Display
    Header.prototype.toggleVersion = function(id) {
        this.versionid = id;
        this.typetext = (this.versionid=='mpl') ? 'MPL' : 'DRP/DAP';

        //switch button title text
        this.resetVersionButtonHtml();
            
        //get current set values & send them
        var verform = this.buildForm();            
        this.sendAjax(verform,'/marvin/setversion/', this.reloadPage);      
    };

    // Reload page
    Header.prototype.reloadPage = function(result) {
        location.reload(true);
    };

    // Send Header Ajax request
    Header.prototype.sendAjax = function(form,url,fxn) {
        $.post($SCRIPT_ROOT + url, form,'json')
        .done(function(data){
            // reload the current page, this re-instantiates a new Header with new version info from session
            if (data.result.status == 1) {
                fxn(data.result);
            } else {
                alert('Failed to set the versions! '+data.result.msg);
            }
        })
        .fail(function(data){
            alert('Failed to set the versions! Problem with Flask setversion. '+data.result.msg);
        });     
    };

    // Toggle Plate/MaNGA ID Search
    Header.prototype.toggleSearchType = function(event) {
        var _this = event.data;
        _this.searchid = $('#idselect option:selected').attr('id');
        _this.searchtext = (_this.searchid == 'plateid') ? 'Plate ID' : 'MaNGA ID';

        $('#idtext').attr('placeholder',_this.searchtext);
        $('#idtext').attr('name',_this.searchid);
        _this.setParams();
        var form = {'searchid':_this.searchid};
        _this.sendAjax(form, '/marvin/setsearch/', _this.setSearchOptions);
    };

    // Set the search options
    Header.prototype.setSearchOptions = function(result) {
        console.log('set search options',result.options);
        if (result.options) {
            var options = {source:result.options};
            $('.typeahead').typeahead('destroy');
            $('.typeahead').typeahead(options);
        }
    };

    // Toggle the mode of Marvin
    Header.prototype.toggleMode = function(event) {
        var _this = event.data;
        _this.marvinmode = $(this).val();
        var modeform = {'marvinmode':_this.marvinmode};
        _this.sendAjax(modeform, '/marvin/setmode/', _this.reloadPage);
    };

    return Header;

})();


