// Javascript code for Marvin DAP QA plots/comments on individual plates
'use strict';

// DAPqa tab selection
$(function() {
    $('#cubetabs a[href*="dapqapane"]').click(function() {
        var ifu = window.ifu.ifu;
        var dapifuform = $('#dapqacomment_form_'+ifu);
        dapifuform.trigger('reset');
        //$('.qacomment').html('');
        var key = 'maps';
        var ready = $('#inspectready').val();

        if (ready === 'true' || ready === 'True') {
            setDefault(ifu,key);
            utils.initTags('#daptagfield');
        } else {
            var fxn = 'loadDapQaPanel';
            $('#fxn').val(fxn);
            $('#dapqapane_'+ifu).hide();
            $('#loginform').modal('show');
        }

        $('#drpcommcollapse_'+ifu).collapse('hide');        
        $('#dapcommcollapse_'+ifu).collapse('show');
    });
});

// Set category default
function setDefault(ifu,key) {
    //var html = (key!='radgrad') ? 'cube-none2': 'rss-rad1';
    var html = (key === 'maps') ? 'cube-none2' : (key ==='spectra') ? 'cube-all5' : 'rss-rad1';
    var mapid = (key === 'maps') ? 'kin' : (key==='radgrad') ? 'emflux': 'spec0';
    $('#dapqacatopts_'+ifu+' #qacomment_'+key).html(html);
    console.log('setdefault', ifu, key, mapid,html);
    console.log($('#dapqacomment_form_'+ifu+' #specpanel'));
    console.log($('#dapqacomment_form_'+ifu+' #specpanel').val());
    $('#dapqacomment_form_'+ifu+' #specpanel').val('single')
    console.log($('#dapqacomment_form_'+ifu+' #specpanel').val());

    // insure correct divs shown
    var mainpanel = $('#dapqapanel_'+ifu);
    if (key === 'maps') {
        $('#dapmap6_'+ifu+'_'+key,mainpanel).show();
        $('#dapmapsingle_'+ifu,mainpanel).hide();        
    } else if (key === 'spectra') {
        $('#dapspecsingle_'+ifu,mainpanel).show();        
        $('#dapmap6_'+ifu+'_'+key,mainpanel).hide();
    }
        
    // display list and panels, store old values
    displayOptions(ifu,key);
    if (key==='spectra') { getSpectraList(ifu,key,mapid,html); }
    getPanel(ifu,key,mapid,html);
    storeold(ifu,key,mapid,html);
    
}


// Toggle DAP categories
function dapcatchange(ifu) {
    // set default selection with new key
    var key = $('#dapqacat_'+ifu).find('#dapqacat_select').val();
    //$('.qacomment').html('');
    setDefault(ifu,key);
}

// Toggle DAP QA cube/rss
$(function() {
    $('.dropdown-menu.qalist').on('click','li a', function() {
        var ifu = window.ifu.ifu;
        var maincat = $('#dapqacatopts_'+ifu);
        var id = $(this).attr('id');
        var parentid = $(this).parent().parent().attr('id');
        var type = parentid.split('typelist')[0];
        var key = parentid.split('_')[1];

        // build html reference
        var html = $('#qacomment_'+key,maincat).html();
        html = type+'-'+id;
        $('#qacomment_'+key,maincat).html(html);
        
        //display the appropriate list
        displayList(ifu,key);

        // get map id
        var mapid = $('#catlist_'+key+' select option:selected',maincat).attr('id');

        console.log('cube/rss toggle', ifu, id,parentid,key, mapid, html);
        //for spectra, populate list for first time
        if (key==='spectra') { getSpectraList(ifu,key,mapid,html); }
        
        //get new panel and store old values
        getPanel(ifu,key,mapid,html);
        storeold(ifu,key,mapid,html);
            
    });
});

// Display options
function displayOptions(ifu,key) {
    var maincat = $('#dapqacatopts_'+ifu);
    var catopts = $('.dapqacatoptions',maincat);
    catopts.hide();
    // mode-bin button
    var singleopts = $('#dapqacatoptions_'+key,maincat);
    singleopts.show();
    // map list
    var catlist = $('.catlist', catopts);
    catlist.hide();
    var singlelist = $('#catlist_'+key,maincat);
    //select first entry and show list
    $('select',singlelist).children().removeProp('selected');
    var first = $('select :first-child',singlelist);
    first.prop('selected',true);
    singlelist.show();
}

// Display the appropriate list and reset selection
function displayList(ifu,key) {
    var maincat = $('#dapqacatopts_'+ifu);
    var catopts = $('.dapqacatoptions',maincat);
    var catlist = $('.catlist', catopts);
    catlist.hide();
    var singlelist = $('#catlist_'+key,maincat);
    singlelist.show();
    $('select',singlelist).children().removeProp('selected');
    
    //set first option as default
    var first = $('select :first-child',singlelist);
    first.prop('selected',true);
}

// Toggle DAP map selection
$(function() {
    $('[id^="dapqacatoptions"]').change(function() {
        var ifu = window.ifu.ifu;
        var maincat = $('#dapqacatopts_'+ifu);
        var id = $(this).attr('id');
        var key = (id.search('map') !== -1) ? 'maps' : (id.search('spectra') !== -1) ? 'spectra' : (id.search('radgrad') !== -1) ? 'radgrad' : '';
        var mapid = $('#'+id+' option:selected',maincat).attr('id');
        var qatype = $('#dapqacatopts_'+ifu+' #qacomment_'+key).html();

        // get old values
        var formcat = $('#dapqacomment_form_'+ifu);
        var oldmapid = $('#oldmapid',formcat).val();
        var oldqatype = $('#oldqatype',formcat).val();
        var oldkey = $('#oldkey',formcat).val();
        console.log('map select id,key,mapid',ifu,id,key,mapid,qatype,oldmapid,oldqatype,oldkey);

        // if mapid is binnum, change to single panel view
        var mainpanel = $('#dapqapanel_'+ifu);
        if (mapid == 'binnum' & oldmapid != 'binnum') {
            $('#dapmap6_'+ifu+'_'+key,mainpanel).hide();
            $('#dapmapsingle_'+ifu,mainpanel).show();
        } else if (mapid !='binnum' & oldmapid == 'binnum') {
            $('#dapmap6_'+ifu+'_'+key,mainpanel).show();
            $('#dapmapsingle_'+ifu,mainpanel).hide();
        }

        // get new panel and store old values
        getPanel(ifu,key,mapid,qatype);
        storeold(ifu,key,mapid,qatype);
    });
});

// Toggle DAP spectrum view
$(function() {
    $('.specview').click(function(){
        var ifu = window.ifu.ifu;
        var formcat = $('#dapqacomment_form_'+ifu);
        var mainpanel = $('#dapqapanel_'+ifu);
        var specviewbut = $('#dapqacatopts_'+ifu+' #toggle_specview');
        if (specviewbut.hasClass('active')) {
            specviewbut.button('reset');
            $('div[id*="dapmap6"]',mainpanel).hide();
            var subpanel = $('div[id*="dapspecsingle"]',mainpanel);
            subpanel.show();
            $('#specpanel',formcat).val('single')
            var oldvals = getold(ifu);
            getPanel(ifu,oldvals.key,oldvals.mapid,oldvals.qatype);
        } else {
            specviewbut.button('complete');
            $('div[id*="dapspecsingle"]',mainpanel).hide();
            var subpanel = $('div[id*="dapmap6"]',mainpanel);
            subpanel.show();
            $('#specpanel',formcat).val('map')            
            var oldvals = getold(ifu);
            getPanel(ifu,oldvals.key,oldvals.mapid,oldvals.qatype);
        }
    });
});

// get current values
function getold(ifu) {
    var formcat = $('#dapqacomment_form_'+ifu);
    var mapid = $('#oldmapid',formcat).val();
    var qatype = $('#oldqatype',formcat).val();
    var key = $('#oldkey',formcat).val();
    return {'key':key,'mapid':mapid,'qatype':qatype};
}

// store old values
function storeold(ifu,key,mapid,qatype) {
    var formcat = $('#dapqacomment_form_'+ifu);
    $('#oldmapid',formcat).val(mapid);
    $('#oldqatype',formcat).val(qatype);
    $('#oldkey',formcat).val(key);
}

// build the DAP form
function buildDAPform(newdata,ifu) {
    var dapform = $('#dapqacomment_form_'+ifu).serializeArray();
    if (newdata) {
        $.each(newdata,function(i,val) {
            dapform.push(val);
        });
    }
    return dapform;
}

// parse DAP issues
function parseDAPissues(ifu,key) {
    var name = '#dapqacomment_form_'+ifu+' select[id*="dapqa_issue_'+key+'"]';
    var issuelist = utils.getSelected(name);
    return issuelist;
}

// get list of DAP spectrum plots available
function getSpectraList(ifu,key,mapid,qatype) {

    var maincat = $('#dapqacatopts_'+ifu);
    var newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'qatype','value':qatype}];
    var dapformdata = buildDAPform(newdata,ifu);
            
    $.post($SCRIPT_ROOT + '/marvin/getdapspeclist', dapformdata,'json')
        .done(function(data){
            
            var speclist = data.result.speclist;
            $('#dapspectralist',maincat).empty();
            if (speclist) {
                $.each(speclist,function(i,name) {
                    
                    var specname = name.split('-')[0];
                    var specnum = name.split('-')[1];
                    specnum = specnum.replace(/^0+/, '');
                    if (specnum.length === 0) {specnum='0';}
                    var id = specname+specnum;
                    
                    $('#dapspectralist',maincat).append($('<option>', {
                        id: id,
                        value: id,
                        text : name,
                        selected : (i===0) ? true : false
                    }));
                });
            }
        })
        .fail(function(){
            $('#dapqa_'+key,maincat).show();
            var title = $('#dapqa_'+key+' h4',maincat);
            var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to retrieve list of spectra!</h4></div>";
            title.html(alerthtml);
        });
}

// get a DAP panel
function getPanel(ifu,key, mapid, qatype) {

    var mainpanel = $('#dapqapanel_'+ifu);
    var mainform = $('#dapqacomment_form_'+ifu);
    var cubepk = $('#'+ifu+' #cubepk').val();
    $('.dapqapanel',mainpanel).hide();
    var specpanel = $('#specpanel',mainform).val();
    
    //grab tags
    var tagbox = $('#daptagfield',mainform).tags();
    var tags = tagbox.getTags();
    tags = JSON.stringify(tags);

    // build form data
    var issues = parseDAPissues(ifu,key);
    var newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'cubepk','value':cubepk},
               {'name':'qatype','value':qatype},{'name':'issues','value':JSON.stringify(issues)},
               {'name':'tags','value':tags}];
    var dapformdata = buildDAPform(newdata,ifu);
    console.log('dapform',dapformdata);
    
    $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,'json')
        .done(function(data){
            var ifupanel = $('#dapqa_'+key,mainpanel);
            ifupanel.show();
            var title = $('#dapqa_'+key+' h4',mainpanel);
            if (data.result.title) {title.html(data.result.title);}
            
            // setsession status failure
            if (data.result.setsession && data.result.setsession.status === -1) {
                var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+data.result.setsession.message+"</h4></div>";
                title.html(alerthtml);
            }

            loadImages(mainpanel,key,mapid,specpanel,data.result,data.result.panelmsg);
            loadComments(mainpanel,key,data.result.getsession);
            loadTags(mainform,data.result.getsession);

            // update count message
            if (data.result.getsession.status === 0) {
                $('#submitmsg',mainform).html("<h5><div class='alert alert-warning' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
            } else if (data.result.getsession.status === 1) {
                $('#submitmsg',mainform).html("<h5><div class='alert alert-info' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
            } else {
                $('#submitmsg',mainform).html("<h5><div class='alert alert-danger' role='alert'>Bad response from inspection database</div></h5>");
            }
        })
        .fail(function(){
            $('#dapqa_'+key,mainpanel).show();
            var title = $('#dapqa_'+key+' h4',mainpanel);
            var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to set session data!</h4></div>";
            title.html(alerthtml);
        });
}

// load DAP plot images
function loadImages(panel,key,mapid,specpanel,results,msg) {
    //$('#dapqa_'+key+' img',panel).removeProp('src'); //seemingly useless, removed to remove the ajax request to unknown
    if (results.images) {

        // select 6 panel or single panel based on map selection (i.e. if binnum)
        if (key != 'spectra') {
            var subpanel = (mapid === 'binnum') ? $('#dapqa_'+key+' div[id*="dapmapsingle"] img',panel) : $('#dapqa_'+key+' div[id*="dapmap6"] img',panel);
        } else {
            var subpanel = (specpanel === 'single') ? $('#dapqa_'+key+' div[id*="dapspecsingle"] img',panel) : $('#dapqa_'+key+' div[id*="dapmap6"] img',panel);
        }

        // load images into panels
        subpanel.each(function(index) {
        	//replace image
            $(this).attr('src',results.images[index]);
            //replace labels
            var labelname = (key !== 'spectra') ? 'Map ' : (specpanel === 'single') ? 'Spectrum ' : 'Line ';
            var labelend = (key !== 'spectra') ? ': '+results.panels[index] : (specpanel === 'map') ? ': '+results.panels[index] : '';
            var labelhtml = labelname+(index+1)+labelend;
            $('#'+key+'label'+(index+1),panel).html(labelhtml);
            //console.log('inside loadImages', specpanel, results.panels[index], labelhtml, panel);
        });
    } else {
        var title = $('#dapqa_'+key+' h4',panel);
        var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+msg+"</h4></div>";
        title.html(alerthtml);
    }
}

// reset and load all the issues and comments
function loadComments(panel,key,results) {
    // reset all panel comments
    $('[id^=dapqa_comment]',panel).val('');
    
    // reset all issue options
    $('[id^=issue]',panel).prop('selected',false);

    // load results if sessioned or stored status
    if (results.status === 0 || results.status === 1) {
    
        // load new comments
        if (results.dapqacomments) {
            $.each(results.dapqacomments,function(i,panelcomment) {

                $('#dapqa_comment'+panelcomment.catid+'_'+panelcomment.position,panel).val(panelcomment.comment);
            
                $.each(panelcomment.issues, function(i,issid) {
                    $('#issue_'+issid+'_'+panelcomment.position,panel).prop('selected',true);
                });
            });
        }
    
    } else {
        // getsession status failure
        var title = $('#dapqa_'+key+' h4',panel);
        var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+results.message+"</h4></div>";
        title.html(alerthtml);
    }
      
    // render the issue multiple selector to update the count
    $('.dapqaissuesp',panel).selectpicker('refresh');
}

// load all the DAP QA current tags + suggestions
function loadTags(panel,results) {
    // set the tag suggestions (autocomplete) to all tags in db ;always run
    var tagbox = $('#daptagfield',panel).tags();
    tagbox.suggestions = results.alltags;
    tagbox.promptText = 'Enter a word or phrase and press Return';
    
    //if none then returns empty list with length = 0
    if (typeof results.tags !== 'undefined' && results.tags.length > 0) {
        jQuery.each(results.tags, function(i,tag) {
            tagbox.addTag(tag);
        });
    }
}

// Load the DAP image modal
function daploadmodal(img) {
    var ifu = window.ifu.ifu;
    var src = $('#'+img).attr('src');
    var name = src.slice(src.search('manga-'));
    $('#dapimgmodal_'+ifu+' #dapimgtitle').html(name);
    var image = '<img class="img-responsive img-rounded" src="'+src+'" alt="Image"/>';
    $('#dapimgmodal_'+ifu+' #dapimgbody').html(image);
}

// Submit/Reset DAP QA Comments
function dapaddcomments(ifu,action) {
    var mainform = $('#dapqacomment_form_'+ifu);
    var maincat =  $('#dapqacatopts_'+ifu);
    var mainpanel = $('#dapqapanel_'+ifu);
    var cubepk = $('#'+ifu+' #cubepk').val();
    var specpanel = $('#specpanel',mainform).val();

    // get current key,mapid,qatype
    var key = $('#dapqacat_'+ifu+' #dapqacat_select').val();
    var qatype = $('#qacomment_'+key,maincat).html();
    var mapid = $('#dap'+key+'list :selected',maincat).attr('id');
    storeold(ifu,key,mapid,qatype);
    
    //grab tags
    var tagbox = $('#daptagfield',mainform).tags();
    var tags = tagbox.getTags();
    tags = JSON.stringify(tags);
        
    // build form data
    var issues = parseDAPissues(ifu,key);
    var newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'cubepk','value':cubepk},
               {'name':'qatype','value':qatype},{'name':'issues','value':JSON.stringify(issues)},{'name':'tags','value':tags},
               {'name':action,'value':true}];
    var dapformdata = buildDAPform(newdata,ifu);

    if (action === "submit") {
        $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,'json')
            .done(function(data){
                var title = $('#dapqa_'+key+' h4',mainpanel);
                if (data.result.title) {title.html(data.result.title);}
                
                // submit message
                if (data.result.setsession) {
                    if (data.result.setsession.status === 0) {
                        $('#submitmsg',mainform).html("<h5><div class='alert alert-warning' role='alert'>"+data.result.setsession.message+"</div></h5>");
                    } else if (data.result.setsession.status === 1) {
                        $('#submitmsg',mainform).html("<h5><div class='alert alert-success' role='alert'>"+data.result.setsession.message+"</div></h5>");
                    } else  {
                        $('#submitmsg',mainform).html("<h4><div class='alert alert-danger' role='alert'>Bad response from inspection module.</div></h4>");
                    }
                }
                
                utils.resetTags('#dapqacomment_form_'+ifu+' #daptagfield');
            })
            .fail(function(){
                var title = $('#dapqa_'+key+' h4',mainpanel);
                var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to get session data!</h4></div>";
                title.html(alerthtml);
            });
    } else if (action === "reset") {
        $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,'json')
            .done(function(data){
                var ifupanel = $('#dapqa_'+key,mainpanel);
                ifupanel.show();
                var title = $('#dapqa_'+key+' h4',mainpanel);
                if (data.result.title) {title.html(data.result.title);}
                
                // setsession status failure
                if (data.result.setsession && data.result.setsession.status === -1) {
                    var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+data.result.setsession.message+"</h4></div>";
                    title.html(alerthtml);
                }
                
                loadImages(mainpanel,key,mapid,specpanel,data.result,data.result.panelmsg);
                loadComments(mainpanel,key,data.result.getsession);
                loadTags(mainform,data.result.getsession);

                // update count message
                if (data.result.getsession.status === 0) {
                    $('#submitmsg',mainform).html("<h5><div class='alert alert-warning' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
                } else if (data.result.getsession.status === 1) {
                    $('#submitmsg',mainform).html("<h5><div class='alert alert-info' role='alert'>"+data.result.getsession.totaldapcomments+"</div></h5>");
                } else {
                    $('#submitmsg',mainform).html("<h5><div class='alert alert-danger' role='alert'>Bad response from inspection database</div></h5>");
                }
                
                utils.resetTags('#dapqacomment_form_'+ifu+' #daptagfield');

            })
            .fail(function(){
                $('#dapqa_'+key,mainpanel).show();
                var title = $('#dapqa_'+key+' h4',mainpanel);
                var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to set session data!</h4></div>";
                title.html(alerthtml);
            });
    }
    
}

// Load the DAP QA Panel
function loadDapQaPanel() {
    var key = 'maps';
    var ifu = window.ifu.ifu;
    $('#dapqapane_'+ifu).show();
    utils.resetLogin();
    setDefault(ifu,key);
}



