// Javascript code for Marvin DAP QA plots/comments on individual plates
'use strict';
    
// DAPqa tab selection
$(function() {
    $('#cubetabs a[href*="dapqapane"]').click(function() {
        var ifu = getIFUHash().slice(1);
        var dapifuform = $('#dapqacomment_form_'+ifu);
        dapifuform.trigger('reset');
        $('.qacomment').html('');
        var key = 'maps';
        var ready = $('#inspectready').val();
        
        if (ready === 'True') {
            setDefault(ifu,key);
            initTags('#daptagfield');
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
    console.log('setdefault', ifu, key, mapid);
    $('#dapqacatopts_'+ifu+' #qacomment_'+key).html(html);
        
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
    $('.qacomment').html('');
    setDefault(ifu,key);
}

// Toggle DAP QA cube/rss
$(function() {
    $('.dropdown-menu.qalist').on('click','li a', function() {
        var ifu = getIFUHash().slice(1);
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
        var ifu = getIFUHash().slice(1);
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

        // get new panel and store old values
        getPanel(ifu,key,mapid,qatype);
        storeold(ifu,key,mapid,qatype);
    });
});

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
    var issuelist = getSelected(name);
    return issuelist;
}

// get list of DAP spectrum plots available
function getSpectraList(ifu,key,mapid,qatype) {

    var maincat = $('#dapqacatopts_'+ifu);
    var newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'qatype','value':qatype}];
    var dapformdata = buildDAPform(newdata,ifu);
            
    $.post($SCRIPT_ROOT + '/marvin/getdapspeclist', dapformdata,null,'json')
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
    
    //grab tags
    var tagbox = $('#daptagfield',mainform).tags();
    var tags = tagbox.getTags();
    tags = JSON.stringify(tags);
        
    // build form data
    var issues = parseDAPissues(ifu,key);
    var newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'cubepk','value':cubepk},
               {'name':'qatype','value':qatype},{'name':'issues','value':JSON.stringify(issues)},{'name':'tags','value':tags}];
    var dapformdata = buildDAPform(newdata,ifu);
    console.log('dapform',dapformdata);

    
    $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,null,'json')
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
            
            loadImages(mainpanel,key,data.result.images,data.result.panelmsg);
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
function loadImages(panel,key,images, msg) {
    $('#dapqa_'+key+' img',panel).removeProp('src');
    if (images) {
        $('#dapqa_'+key+' img',panel).each(function(index) {
            $(this).attr('src',images[index]);
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

    // load results if good status
    if (results.status === 1) {
    
        // load new comments
        if (results.dapqacomments) {
            $.each(results.dapqacomments,function(i,panelcomment) {
        
                //console.log('panelcomment',panelcomment);
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
    
    //if none then returns empty list with length = 0
    if (typeof results.tags !== 'undefined' && results.tags.length > 0) {
        jQuery.each(results.tags, function(i,tag) {
            tagbox.addTag(tag);
        });
    }
}

// Load the DAP image modal
function daploadmodal(img) {
    var ifu = getIFUHash().slice(1);
    var src = img.src;
    var name = src.slice(src.search('manga-'));
    $('#dapimgmodal_'+ifu+' #dapimgtitle').html(name);
    var image = '<img class="img-responsive img-rounded" src="'+src+'" alt="Image"/>';
    $('#dapimgmodal_'+ifu+' #dapimgbody').html(image);
}

// Submit DAP QA Comments
function dapaddcomments(ifu) {
    var mainform = $('#dapqacomment_form_'+ifu);
    var maincat =  $('#dapqacatopts_'+ifu);
    var mainpanel = $('#dapqapanel_'+ifu);
    var cubepk = $('#'+ifu+' #cubepk').val();

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
               {'name':'submit','value':true}];
    var dapformdata = buildDAPform(newdata,ifu);

    $.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,null,'json')
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
            
        })
        .fail(function(){
            var title = $('#dapqa_'+key+' h4',mainpanel);
            var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to get session data!</h4></div>";
            title.html(alerthtml);
        });
}

// Reset the DAP QA form
function resetDAPQA(ifu) {
    //var maincat =  $('#dapqacatopts_'+ifu);
    var mainpanel = $('#dapqapanel_'+ifu);
    //$('.qacomment',maincat).html('');
    //mainform.trigger('reset');
    //$('.dapqacatoptions',maincat).hide();
    //$('.catlist',maincat).hide();
    $('[id^=dapqa_comment]',mainpanel).val('');
    $('[id^=issue]',mainpanel).prop('selected',false);
    $('.dapqaissuesp',mainpanel).selectpicker('refresh');
    resetTags('#dapqacomment_form_'+ifu+' #daptagfield');
}

// Load the DAP QA Panel
function loadDapQaPanel() {
    var key = 'maps';
    var ifu = getIFUHash().slice(1);
    $('#dapqapane_'+ifu).show();
    resetLogin();
    setDefault(ifu,key);
}



