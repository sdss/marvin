// Javascript code for Marvin comments on individual plates
'use strict';

    // Toggle between comment form categories
    $(function() {        
        $('#commentcat').change(function () {
            // hide everything
            $('.catissues').hide();
            $('.catcomments').hide();
            // display new things
            $('#catcomment_'+$(this).val()).fadeIn();
            var catvalue = $('#commentcat option:selected').val();
            if (catvalue === 'dapqa') {$('.dapqacomms').show();}
            // deal with issues
            console.log('catvalue',catvalue);
            $('#catissue_'+$(this).val()).fadeIn();
        });
    });
    
    // Reset comment form
    function resetCommentForm() {
        $('#commentform').modal('hide');
        $('#addcomment_form').trigger("reset");
        $('#commentmessage').empty();
        $('.catissues').hide();
        $('#catissue_'+$('#commentcat').val()).fadeIn();
        $('.catcomments').hide();
        $('#catcomment_'+$('#commentcat').val()).fadeIn();
        resetTags('#tagfield');
        /*var tagbox = $('#tagfield').tags();
        var tags = tagbox.getTags();
        jQuery.each(tags, function(i,tag) {
            tagbox.removeTag(tag);
        });
        tagbox.removeLastTag();*/
    }

    // Submit add comment form
    function submitcomment() {

        // grab cube info
        var plateid = $('#plate').val();
        var version = $('#drpver').val();        
        var cubepk = $('#cubepk').val();
        var ifuname = $('#ifuname').val(); 
                
        // grab comments from all boxes
        var comments = [];
        $('.commentfield').each(function(index){
            comments.push(this.value);
        });
        comments = JSON.stringify(comments);

        // grab all checked issues
        var issueids = [];
          var issuelist = getSelected('.issuesp');
          if (typeof issuelist === Array) {
            $.each(issuelist, function (index,value) {
                console.log(typeof value, value.length);
                issueids.push(parseInt(value.split('issue')[1]));
            });
          }
          issueids = JSON.stringify(issueids);

        //grab tags
        var tagbox = $('#tagfield').tags();
        var tags = tagbox.getTags();
        tags = JSON.stringify(tags);

        $.post($SCRIPT_ROOT + '/marvin/addcomment', {'cubepk':cubepk,'plateid':plateid,'ifuname':ifuname,'version':version,
            'comments':comments,'issueids':issueids, 'tags':tags},'json')
            .done(function(data){
                if (data.result.status < 0) {
                    // bad submit
                    resetCommentForm();
                } else {
                    // good response from Inspection, comment ignored with attention message
                    if (data.result.message !== ''){
                        var stat = (data.result.status === 0) ? 'danger' : 'success'; 
                        var htmlstr = "<div class='alert alert-"+stat+"' role='alert'><h4>" + data.result.message + "</h4></div>";
                        $('#commentmessage').html(htmlstr);
                    }
                    // good response from Inspection, comment submitted with success
                    if (data.result.status === 1){
                        setTimeout(populateComments, 1500, data);
                        setTimeout(resetCommentForm, 1500);
                    }
                }
            })
            .fail(function(){
                alert("Error in response from Inspection webapp.  Please contact admin@sdss.org");
            });    

    }
    
    // Initialize comment tags
    initTags('#tagfield');
    /*$(function() {
        var tagbox = $('#tagfield').tags({
            tagData:[],
            tagSize:'sm',
            suggestions:[],
            caseInsensitive: true
        });        
    });*/
    
    // Get previous comments on cube from specific user
    function getComment(cubepk, ifuname) {
    
        // grab and set cube info
        var plateid = $('#plate').val();
        var version = $('#drpver').val();
        $('#cubepk').val(cubepk);
        $('#ifuname').val(ifuname);
                                                        
        $.post($SCRIPT_ROOT + '/marvin/getcomment', {'cubepk':cubepk,'plateid':plateid,'ifuname':ifuname,'version':version},'json') 
            .done(function(data){
                // show commentform if ready (login **may be** necessary)
                if (data.result.ready) {
                    $('#commentform').modal('show');
                    populateComments(data);
                    populateTags(data);
                } else {
                    $('#loginform').modal('show');
                }
            })
            .fail(function(){
            });                            
    }
    
    // Populate comment fields with prior comments
    function populateComments(data) {
        $('.catcomments').each(function(index){
            
            var userlabel = $('#userlabel_'+(index+1)).text();
            try {
                var comment = data.result.comments[index+1];
                var comment_comment = comment.comment;
                var comment_issues = comment.issues;
                var comment_modified = comment.modified;
                if (!!comment_modified) {
                    comment_modified = "Submitted on " + comment_modified;
                }
            }
            catch(err) {
                var comment_comment = '';
                var comment_issues = [];
                var comment_modified = '';
                //var comment_modified = err;
            }
            var recentcomments = data.result.recentcomments[index+1];
            var htmlstr="<option value='comment0' value='' selected></option>";
            
            $.each(recentcomments, function(i,value) {
            	htmlstr += '<option value="comment'+(i+1)+'">'+value+'</option>';
            });
            
            if (userlabel.search(data.result.membername) < 0) {
                userlabel += data.result.membername;
            }
            $('#userlabel_'+(index+1)).text(userlabel);
            $('#recentcomments_'+(index+1)).html(htmlstr);
            $('#commentfield_'+(index+1)).html(comment_comment);
            $('#commentsubmitted_'+(index+1)).html(comment_modified);
            try {
                $.each(comment_issues, function(i,value) {
                	$('#issue'+value).prop('selected',true);
                });
            }
            catch(err) {
                $('#commentsubmitted_'+(index+1)).html(err);
            }
            $('.selectpicker').selectpicker('refresh');

        });
    }        
    
    //Populate the comment box with current tags + suggestions
    function populateTags(data) {
        // set the tag suggestions (autocomplete) to all tags in db ; always run
        var tagbox = $('#tagfield').tags();
        tagbox.suggestions = data.result.alltags;
        
        //if none then returns empty list with length = 0
        if (data.result.tags.length > 0) {
            jQuery.each(data.result.tags, function(i,tag) {
                tagbox.addTag(tag);
            });
        }
        console.log('alltags',data.result.alltags);
        console.log('tags',data.result.tags);
        console.log('assigned tags',tagbox.getTags());
    }
    
    // Input selected comment option into new comment 
    $(function(){
        $('.usercomments').change(function() {
            var index = $(this).attr('id').search('_');
            var id = $(this).attr('id').slice(index+1);
            
            var selectedcomment = $('#recentcomments_'+id+' option:selected').text();
            $('#commentfield_'+id).text(selectedcomment);
        });
    });

    // Grab the comments after a successful login (function to run after login)
    function grabComments(result) {
        var cubepk = $('#cubepk').val();
        var ifuname = $('#ifuname').val();
        resetLogin();
        getComment(cubepk, ifuname);
    }