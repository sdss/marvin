// Javascript code for DAP QA 
'use strict';

var Dapqa,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Dapqa = (function() {

    // Constructor
    function Dapqa(ifu) {

        // in case constructor called without new
        if (false === (this instanceof Dapqa)) {
            return new Dapqa();
        }
        
        this.init(ifu);

        // Event handlers
        // on DAP Tab selection
        this.tabselect.on('click',this,this.toggleTab);
        // on DAP category change
        this.catselect.on('change',this,this.toggleCategory);
        // on DAP cube/RSS options select 
        this.optsselect.on('click','li a', this, this.toggleOptions);
        // on DAP map list change
        //$('[id^="dapqacatoptions"]',this.maincat).on('change',this,this.toggleMap);
        this.generalmaps.on('change',this,this.toggleMap);
        // on DAP spectral button click
        this.specviewbut.on('click', this, this.toggleSpecView);
        
    }

    // initialize the object
    Dapqa.prototype.init = function(ifu) {
        // parameters
        this.ifu = ifu;
        this.selectedtab = $('#cubetabs a[href*="samplepane"]').attr('id');
        this.tabhash = '#'+this.selectedtab;
        this.cubepk = $('#'+this.ifu+' #cubepk').val();
        this.key = 'maps';
        this.mapid = null;
        this.qatype = null;
        this.oldkey = null;
        this.oldmapid = null;
        this.oldqatype = null;
        this.specpaneltype = $('#specpanel', this.mainpanel).val();
        this.mappaneltype = null;
        this.oldspecpanel = this.specpaneltype;
        this.ready = $('#inspectready').val();
        this.fxn = 'loadDapQaPanel';
        this.optionstype = null;
        this.optionsid = null;
        this.issues = null;
        this.tags = null;

        // div elements for various pieces of the DAP QA tab, for a given IFU
        this.tabselect = $('#cubetabs a[href*="dapqapane_'+this.ifu+'"]'); // dapqa tab panel select element
        this.maintab = $('#dapqapane_' + this.ifu); // main dapqa tab element
        this.mainform = $('#dapqacomment_form_' + this.ifu); // main dapqa form element
        this.catselect = $('#dapqacat_' + this.ifu).find('#dapqacat_select'); //category select 
        this.mainpanel = $('#dapqapanel_' + this.ifu); // main dap panel for plots
        this.maincat = $('#dapqacatopts_' + this.ifu); // main panel for dap category options
        this.ifupanel = $('#dapqa_' + this.key, this.mainpanel); // ifu panel
        this.paneltitle = $('h4', this.ifupanel) // title area on DAP map panel
        this.qaoptionstext = $('#qacomment_'+ this.key, this.maincat); // dap qa options displayed html text
        this.catopts = $('.dapqacatoptions',this.maincat); // all sub-divs for category options
        this.singlecatopts = $('#dapqacatoptions_'+ this.key,this.maincat); // individual div for dap category options
        this.optsselect = $('.dropdown-menu.qalist', this.maincat); // DAP cube/rss options button select
        this.catlist = $('.catlist', this.catopts); // all sub-divs for the list of map types
        this.singlecatlist = $('#catlist_'+this.key,this.maincat); // individual div for specific dap map list
        this.maplist = $('#dap'+this.key+'list',this.singlecatlist); // actual map list select block
        this.generalmaps = $('[id^="dapqacatoptions"]',this.maincat); // general options, used for map list change (can't get specific ones to work)
        this.specviewbut = $('#toggle_specview', this.maincat); // toggle button between spectral single vs 6 map panel
        this.subpanel = null; //subpanel for map/spectral view
        this.drpcomments = $('#drpcommcollapse_' + this.ifu); // collapsible drp comment table
        this.dapcomments = $('#dapcommcollapse' + this.ifu); // collapsible dap comment table
        this.map6panel = $('#dapmap6_'+this.ifu+'_'+this.key,this.mainpanel); // div for dap 6 panel map/spectra view
        this.mapsinglepanel = $('#dapmapsingle_'+this.ifu,this.mainpanel) // div for dap single panel map view
        this.specsinglepanel = $('#dapspecsingle_'+this.ifu,this.mainpanel); // div for dap single spectral view
        this.imagemodal = $('#dapimgmodal_'+this.ifu); // DAP image modal (for zoom-in)
        this.imagemodaltitle = $('#dapimgtitle', this.imagemodal); // DAP image modal title
        this.imagemodalbody = $('#dapimgbody', this.imagemodal); // DAP image modal body

        //this.tagbox = $('#daptagfield', this.mainform); // DAP tag box
        this.tagname = '#dapqacomment_form_' + this.ifu + ' #daptagfield';
        this.tagbox = utils.initTags(this.tagname);

    };

    // test print
    Dapqa.prototype.print = function() {
        console.log('We are now printing dapqa info: ', this.ifu, this.selectedtab);
    };

    // load the DAP QA Panel
    Dapqa.prototype.loadDapQaPanel = function loadDapQaPanel() {
        this.key = 'maps';
        this.maintab.show();
        utils.resetLogin();
        this.ready = $('#inspectready').val();
        this.setDefault();
    };

    // Toggle DAP QA tab selection
    Dapqa.prototype.toggleTab = function(event) {
        var _this = event.data;
        _this.mainform.trigger('reset');

        // set new tab
        _this.selectedtab = $('#cubetabs a[href*="dapqapane"]').attr('id');
        _this.tabhash = '#'+_this.selectedtab;

        // load panel or log in
        if (_this.ready === 'true' || _this.ready === 'True') {
            _this.setDefault(_this.ifu,_this.key);
        } else {
            $('#fxn').val(_this.fxn);
            utils.setFunction(_this.loadDapQaPanel, _this);
            _this.maintab.hide();
            $('#loginform').modal('show');
        }

        // show existing dap comments
        _this.drpcomments.collapse('hide');        
        _this.dapcomments.collapse('show');
    };

    // Toggle DAP category change
    Dapqa.prototype.toggleCategory = function(event) {
        var _this = event.data;
        // set default selection with new key
        _this.key = _this.catselect.val();
        _this.setDefault();
    };

    // Set the default DAP QA values
    Dapqa.prototype.setDefault = function() {
        // set some new params
        this.qatype = (this.key === 'maps') ? 'cube-none2' : (this.key ==='spectra') ? 'cube-all5' : 'rss-rad1';
        this.mapid = (this.key === 'maps') ? 'kin' : (this.key==='radgrad') ? 'emflux': 'spec0';
        this.optionstype = this.qatype.split('-')[0];
        this.optionsid = this.qatype.split('-')[1];
        this.specpaneltype = 'single';
        this.oldspecpanel = this.specpaneltype;
        this.mappaneltype = 'map';
        $('#specpanel', this.mainpanel).val(this.specpaneltype);

        // set new panels with new key
        this.singlecatopts = $('#dapqacatoptions_'+ this.key,this.maincat);
        this.singlecatlist = $('#catlist_'+this.key,this.maincat);
        this.maplist = $('#dap'+this.key+'list',this.singlecatlist);
        this.map6panel = $('#dapmap6_'+this.ifu+'_'+this.key,this.mainpanel);
        this.qaoptionstext = $('#qacomment_'+ this.key, this.maincat);
        this.qaoptionstext.html(this.qatype);
        this.ifupanel = $('#dapqa_' + this.key, this.mainpanel);
        this.paneltitle = $('h4', this.ifupanel);
        this.getSubPanel();

        // insure correct divs shown for maps/spectra (6-panel or single panel)
        if (this.key === 'maps') {
            this.map6panel.show();
            this.mapsinglepanel.hide();
            this.mappaneltype='map';        
        } else if (this.key === 'spectra') {
            this.specsinglepanel.show();        
            this.map6panel.hide();
        }
        
        // display list and panels, store old values
        this.displayOptions();
        if (this.key==='spectra') { this.getSpectraList(); }
        this.getPanel();
        this.storeOldValues();

    };

    // Display Category Options
    Dapqa.prototype.displayOptions = function() {
        this.catopts.hide();
        // mode-bin button
        this.singlecatopts.show();
        // map list
        this.catlist.hide();
        //select first entry and show list
        $('select',this.singlecatlist).children().removeProp('selected');
        var first = $('select :first-child',this.singlecatlist);
        first.prop('selected',true);
        this.singlecatlist.show();
    };

    // Display Map Lists
    Dapqa.prototype.displayList = function() {
        this.catlist.hide();
        this.singlecatlist.show();
        $('select',this.singlecatlist).children().removeProp('selected');
        
        //set first option as default
        var first = $('select :first-child',this.singlecatlist);
        first.prop('selected',true);        
    };

    // Toggle DAP Cube/RSS options
    Dapqa.prototype.toggleOptions = function(event) {
        var _this = event.data;
        var parentid = $(this).parent().parent().attr('id');
        _this.optionsid = $(this).attr('id');
        _this.optionstype = parentid.split('typelist')[0];
        _this.key = parentid.split('_')[1];

        // build qatype reference
        _this.qatype = _this.optionstype+'-'+_this.optionsid;
        _this.qaoptionstext.html(_this.qatype);
        
        //display the appropriate list and get the map id
        _this.displayList();
        _this.mapid = $('select option:selected', _this.singlecatlist).attr('id');

        //console.log('cube/rss toggle', _this.ifu, _this.optionstype, _this.optionsid, _this.key, _this.mapid, _this.qatype);

        //for spectra, populate list for first time
        if (_this.key==='spectra') { _this.getSpectraList(); }
        
        //get new panel and store old values
        _this.getOldValues();
        _this.getPanel();
        _this.storeOldValues();        
    };

    // Toggle the DAP Map
    Dapqa.prototype.toggleMap = function(event) {
        var _this = event.data;
        var id = $(this).attr('id');
        _this.mapid = $('#'+id+' option:selected', _this.maincat).attr('id');
        _this.qatype = _this.qaoptionstext.html();

        // get old values
        _this.getOldValues();
        //console.log('map select id,key,mapid',_this.ifu, id, _this.key, _this.mapid, _this.qatype);
        //console.log('old select id,key,mapid',_this.ifu, id, _this.oldkey, _this.oldmapid, _this.oldqatype);

        // if mapid is binnum, change to single panel view
        if (_this.mapid == 'binnum' & _this.oldmapid != 'binnum') {
            _this.map6panel.toggle();
            _this.mapsinglepanel.toggle();
            _this.mappaneltype = 'single';
        } else if (_this.mapid !='binnum' & _this.oldmapid == 'binnum') {
            _this.map6panel.toggle();
            _this.mapsinglepanel.toggle();
            _this.mappaneltype = 'map';
        }

        // get new panel and store old values
        _this.getPanel();
        _this.storeOldValues();        
    };

    // Toggle the DAP Spectral View (6 Panel or Single)
    Dapqa.prototype.toggleSpecView = function(event) {
        var _this = event.data;
        _this.oldspecpanel = _this.specpaneltype;
        if (_this.specviewbut.hasClass('active')) {
            _this.specviewbut.button('reset');
            $('div[id*="dapmap6"]', _this.mainpanel).toggle();
            _this.subpanel = $('div[id*="dapspecsingle"]', _this.mainpanel);
            _this.subpanel.toggle();
            _this.specpaneltype = 'single';
        } else {
            _this.specviewbut.button('complete');
            $('div[id*="dapspecsingle"]', _this.mainpanel).toggle();
            _this.subpanel = $('div[id*="dapmap6"]', _this.mainpanel);
            _this.subpanel.toggle();
            _this.specpaneltype = 'map';
        }
        $('#specpanel',_this.mainform).val(_this.specpaneltype);        
        $('#oldspecpanel', _this.mainform).val(_this.oldspecpanel);
        _this.getOldValues();
        _this.getPanel();
        _this.storeOldValues();
    };

    // Validate DAP form
    Dapqa.prototype.validateForm = function(form) {
        var tmp = [];
        $.each(form, function(i, param) {
            // everything should be a string 
            if (typeof param.value !== 'string') {
                throw new Error('Error validating form: Parameter '+param.name+' with value '+param.value+' is not a string');
            }
            // should be no duplicate entry names in form
            if ($.inArray(param.name, tmp) !== -1) {
                throw new Error('Error validating form: Duplicate name in form for '+param.name);
            }
            tmp.push(param.name);

        });
    };

    // Build DAP Form
    Dapqa.prototype.buildDapForm = function(newdata) {
        // get existing form data
        var dapform = this.mainform.serializeArray();
        // adding new components
        if (newdata) {
            $.each(newdata,function(i,val) {
                dapform.push(val);
            });
        }

        //validate form
        try {
            this.validateForm(dapform);
        } catch (error) {
            console.error('Error building DAP form: '+error);
            Raven.captureException('Error building DAP form: '+error);
        }

        return dapform;
    };

    // Get new form data
    Dapqa.prototype.getNewFormData = function() {
        var newdata = [{'name':'key','value':this.key},{'name':'mapid','value':this.mapid},{'name':'cubepk','value':this.cubepk},
                   {'name':'qatype','value':this.qatype},{'name':'issues','value':this.issues},
                   {'name':'tags','value': this.tags}, {'name':'mappanel','value':this.mappaneltype}];
        return newdata;
    };

    // Validate issues
    Dapqa.prototype.validateIssues = function(issues) {
        var _this = this;
        // issues is not an array or a string value of 'any'
        if (typeof issues !== Array && typeof issues !== 'object' && issues !== 'any') {
            throw new Error('Error validating issues: '+issues+' is not an array or any');
        }
        // any element of issue array is not in correct format
        if (typeof issues == 'object') {
            $.each(issues, function(index, value) {
                // not splittable into 3
                var tmpval = value.split('_');
                if (tmpval.length !== 3) throw new Error('Error validating issues: '+value+' element not splittable; does not have correct format');
                // 1st element != "issue"
                if (tmpval[0] !== 'issue') throw new Error('Error validating issues: 1st element of '+value+' not issue');
                // 2nd element not a number
                if ($.isNumeric(tmpval[1]) == false) throw new Error('Error validating issues: 2nd element of '+value+' not a number');
                // 3rd element not a number between 1-6 or not a string of binnum or single
                if ($.isNumeric(tmpval[2]) == false) {
                    if (tmpval[2] !== 'binnum' && tmpval[2] !== 'single') throw new Error('Error validating issues: 3rd element of '+value+' not string binnum or single');
                } else if (tmpval[2] < 1 || tmpval[2] > 6) {
                    throw new Error('Error validating issues: 3rd element of '+value+' not a number or outside range 1-6');
                }
            });
        }
    };    

    // Parse the DAP Issues
    Dapqa.prototype.parseDapIssues = function(element) {
        //var name = '#dapqacomment_form_'+this.ifu+' select[id*="dapqa_issue_'+this.key+'"]';
        var name = $('select[id*="dapqa_issue_'+this.key+'"]', element);

        var issuelist = utils.getSelected(name);
        // remove duplicate entries        
        var issuelist = (typeof issuelist == 'object') ? utils.unique(issuelist) : issuelist;
 
        // try issue validation
        try {
            this.validateIssues(issuelist);
        } catch (error) {
            issuelist = 'any';
            console.error('Error in getPanel: '+error);
            Raven.captureException('Error in getPanel: '+error);
        }

        this.issues = JSON.stringify(issuelist);
    };

    // Store old values for key,mapid,qatype in the form
    Dapqa.prototype.storeOldValues = function() {
        $('#oldmapid',this.mainform).val(this.mapid);
        $('#oldqatype',this.mainform).val(this.qatype);
        $('#oldkey',this.mainform).val(this.key);
        $('#oldspecpanel',this.mainform).val(this.specpaneltype);        
    }

    // Get old values for key, mapid, qatype from the form
    Dapqa.prototype.getOldValues = function() {
        this.oldmapid = $('#oldmapid',this.mainform).val();
        this.oldqatype = $('#oldqatype',this.mainform).val();
        this.oldkey = $('#oldkey',this.mainform).val();
        this.oldspecpanel = $('#oldspecpanel',this.mainform).val();
    }

    // Get the appropriate subpanel
    Dapqa.prototype.getSubPanel = function() {
        if (this.key != 'spectra') {
            this.subpanel = (this.mapid === 'binnum') ? $('#dapqa_'+this.key+' div[id*="dapmapsingle"] img', this.mainpanel) : $('#dapqa_'+this.key+' div[id*="dapmap6"] img',this.mainpanel);
        } else {
            this.subpanel = (this.specpaneltype === 'single') ? $('#dapqa_'+this.key+' div[id*="dapspecsingle"] img', this.mainpanel) : $('#dapqa_'+this.key+' div[id*="dapmap6"] img',this.mainpanel);
        }
    };

    // Validate tags
    Dapqa.prototype.validateTags = function(tags) {
        // tags is not an array
        if (typeof tags !== Array && typeof tags !== 'object') {
            throw new Error('Error validating tags: '+tags+' is not an array');
        }
    };

    // Grab DAP tags
    Dapqa.prototype.grabTags = function() {
        this.tags = this.tagbox.getTags();

        // try tag validation
        try {
            this.validateTags(this.tags);
        } catch (error) {
            console.error('Error in grabTags: '+error);
            Raven.captureException('Error in grabTags: '+error);
        }

        this.tags = JSON.stringify(this.tags);
    };

    // Retrieve the list of spectral plots available
    Dapqa.prototype.getSpectraList = function() {
        var newdata = [{'name':'key','value':this.key},{'name':'mapid','value':this.mapid},{'name':'qatype','value':this.qatype}];
        var dapformdata = this.buildDapForm(newdata);
        var _this = this;
                
        $.post($SCRIPT_ROOT + '/marvin/getdapspeclist', dapformdata,'json')
            .done(function(data){
                
                var speclist = data.result.speclist;
                $('#dapspectralist', _this.maincat).empty();
                if (speclist) {
                    $.each(speclist,function(i,name) {
                        
                        var specname = name.split('-')[0];
                        var specnum = name.split('-')[1];
                        specnum = specnum.replace(/^0+/, '');
                        if (specnum.length === 0) {specnum='0';}
                        var id = specname+specnum;
                        
                        $('#dapspectralist', _this.maincat).append($('<option>', {
                            id: id,
                            value: id,
                            text : name,
                            selected : (i===0) ? true : false
                        }));
                    });
                }
            })
            .fail(function(){
                $('#dapqa_'+ _this.key, _this.maincat).show();
                var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to retrieve list of spectra!</h4></div>";
                _paneltitle.html(alerthtml);
            });
    };

    Dapqa.prototype.isVisible = function(element) {
        return element.css('display') != 'none';
    };

    Dapqa.prototype.getVisibleChildren = function(element) {
        return element.children('div:not([style*="none"])');
    };

    // Retrieve the DAP map panels
    Dapqa.prototype.getPanel = function() {
        var _this = this;
        $('.dapqapanel', this.mainpanel).hide();
        this.specpaneltype = $('#specpanel', this.mainform).val();
        var visiblechildren  = this.getVisibleChildren(this.ifupanel);
        
        // grab tags
        this.grabTags();

        // parse DAP issues
        this.parseDapIssues(this.ifupanel);

        // build form data
        var newdata = this.getNewFormData();
        var dapformdata = this.buildDapForm(newdata);
        console.log('dapform',dapformdata);
        
        $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,'json')
            .done(function(data){
                _this.ifupanel.show();
                if (data.result.title) {_this.paneltitle.html(data.result.title);}
                
                // setsession status failure
                if (data.result.setsession && data.result.setsession.status === -1) {
                    var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+data.result.setsession.message+"</h4></div>";
                    _this.paneltitle.html(alerthtml);
                }

                if (data.result.status === 1) {
                    _this.loadImages(data.result,data.result.panelmsg);
                    _this.loadComments(data.result.getsession);
                    _this.loadTags(data.result.getsession);

                    // update count message
                    if (data.result.getsession.status === 0) {
                        $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-warning' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
                    } else if (data.result.getsession.status === 1) {
                        $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-info' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
                    } else {
                        $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-danger' role='alert'>Bad response from inspection database</div></h5>");
                    }

                } else {
                    var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+data.result.panelmsg+"</h4></div>";
                    _this.paneltitle.html(alerthtml);
                    $('img',_this.ifupanel).attr('src','');
                    _this.resetCommentsAndIssues();                    
                }

            })
            .fail(function(data){
                $('#dapqa_'+ _this.key, _this.mainpanel).show();
                var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to get/set session data! </h4></div>";
                _this.paneltitle.html(alerthtml);
            });
    };

    // Load DAP Images 
    Dapqa.prototype.loadImages = function(results,msg) {
        // reset previous src link from images
        $('img',this.ifupanel).attr('src','');
        // load new images
        if (results.images) {
            // select 6 panel or single panel based on map selection (i.e. if binnum)
            this.getSubPanel();

            // load images into panels
            var _this = this;
            this.subpanel.each(function(index) {
                //replace image & labels
                $(this).attr('src',results.images[index]);
                var labelname = (_this.key !== 'spectra') ? 'Map ' : (_this.specpaneltype == 'single') ? 'Spectrum ' : 'Line ';
                var labelend = (_this.key !== 'spectra') ? ': '+results.panels[index] : (_this.specpaneltype == 'map') ? ': '+results.panels[index] : '';
                var labelhtml = labelname+(index+1)+labelend;
                var label = _this.getVisibleChildren(_this.ifupanel).find('#'+_this.key+'label'+(index+1));
                label.html(labelhtml);
            });
        } else {
            var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+msg+"</h4></div>";
            this.paneltitle.html(alerthtml);
        }
    };

    // Load the DAP comments + issues
    Dapqa.prototype.loadComments = function(results) {
        var _this = this;
        console.log('loading comments',results.dapqacomments, results.status);
        // reset all panel comments
        $('[id^=dapqa_comment]', this.mainpanel).val('');
        
        // reset all issue options
        $('[id^=issue]', this.mainpanel).prop('selected',false);

        // load results if sessioned or stored status
        if (results.status === 0 || results.status === 1) {
        
            // load new comments
            if (results.dapqacomments) {
                $.each(results.dapqacomments,function(i,panelcomment) {

                    //var position = (_this.mapid=='binnum') ? 'binnum' : (_this.key=='spectra' && _this.specpaneltype=='single') ? 'single' : panelcomment.position;

                    $('#dapqa_comment'+panelcomment.catid+'_'+panelcomment.position, _this.mainpanel).val(panelcomment.comment);
                
                    $.each(panelcomment.issues, function(i,issid) {
                        $('#issue_'+issid+'_'+panelcomment.position, _this.mainpanel).prop('selected',true);
                    });
                });
            }
        
        } else {
            // getsession status failure
            var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+results.message+"</h4></div>";
            this.paneltitle.html(alerthtml);
        }
          
        // render the issue multiple selector to update the count
        $('.dapqaissuesp', this.mainpanel).selectpicker('refresh');
    };

    // Load the DAP current tags + suggestions
    Dapqa.prototype.loadTags = function(results) {
        // set the tag suggestions (autocomplete) to all tags in db ;always run
        var _this = this;
        this.tagbox.suggestions = results.alltags;
        this.tagbox.promptText = 'Enter a word or phrase and press Return';
        
        //if none then returns empty list with length = 0
        if (typeof results.tags !== 'undefined' && results.tags.length > 0) {
            jQuery.each(results.tags, function(i,tag) {
                _this.tagbox.addTag(tag);
            });
        }
    };

    // Load the DAP Image Modal
    Dapqa.prototype.loadImageModal = function(img) {
        var src = $('#'+img).attr('src');
        var name = src.slice(src.search('manga-'));
        this.imagemodaltitle.html(name);
        var image = '<img class="img-responsive img-rounded" src="'+src+'" alt="Image"/>';
        this.imagemodalbody.html(image);
    };

    // Reset DAP comments and issues
    Dapqa.prototype.resetCommentsAndIssues = function() {
        $('[id^=dapqa_comment]', this.mainpanel).val('');        
        $('[id^=issue]', this.mainpanel).prop('selected',false);
        $('.dapqaissuesp', this.mainpanel).selectpicker('refresh');
    };

    // Submit or Reset DAP Comments
    Dapqa.prototype.addComments = function(action) {
        console.log('adding comments',action);

        var _this = this;
        this.storeOldValues();
        this.grabTags(); //grab tags
        this.parseDapIssues(this.ifupanel); //parse issues    

        // build form data
        var newdata = this.getNewFormData();
        var dapformdata = this.buildDapForm(newdata);

        // perform the Ajax request based on the action
        if (action === 'submit') {
            $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,'json')
                .done(function(data){
                    if (data.result.title) {_this.paneltitle.html(data.result.title);}
                    
                    // submit message
                    if (data.result.setsession) {
                        if (data.result.setsession.status === 0) {
                            $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-warning' role='alert'>"+data.result.setsession.message+"</div></h5>");
                        } else if (data.result.setsession.status === 1) {
                            $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-success' role='alert'>"+data.result.setsession.message+"</div></h5>");
                        } else  {
                            $('#submitmsg', _this.mainform).html("<h4><div class='alert alert-danger' role='alert'>Bad response from inspection module.</div></h4>");
                        }
                    }
                    //reset tags
                    utils.resetTags(_this.tagname);
                })
                .fail(function(){
                    var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to get session data!</h4></div>";
                    _this.paneltitle.html(alerthtml);
                });
        } else if (action === 'reset') {
            $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,'json')
                .done(function(data){
                    _this.ifupanel.show();
                    if (data.result.title) {_this.paneltitle.html(data.result.title);}
                    
                    // setsession status failure
                    if (data.result.setsession && data.result.setsession.status === -1) {
                        var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+data.result.setsession.message+"</h4></div>";
                        _this.paneltitle.html(alerthtml);
                    }
                    
                    _this.loadImages(data.result,data.result.panelmsg);
                    _this.loadComments(data.result.getsession);
                    _this.loadTags(data.result.getsession);

                    // update count message
                    if (data.result.getsession.status === 0) {
                        $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-warning' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
                    } else if (data.result.getsession.status === 1) {
                        $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-info' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
                    } else {
                        $('#submitmsg', _this.mainform).html("<h5><div class='alert alert-danger' role='alert'>Bad response from inspection database</div></h5>");
                    }
                    
                    utils.resetTags(_this.tagname);

                })
                .fail(function(){
                    $('#dapqa_'+ _this.key, _this.mainpanel).show();
                    var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to set session data!</h4></div>";
                    _this.paneltitle.html(alerthtml);
                });
        }

    };

    return Dapqa;
})();
