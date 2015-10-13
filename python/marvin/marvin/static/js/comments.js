'use strict';

var Comment,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Comment = (function () {

    function Comment(ifu) {

        // in case constructor called without new
        if (false === (this instanceof Comment)) {
            return new Comment();
        }
        
        this.init(ifu);
        
        // Event Handlers
        // toggle new comment category on change
        $('#commentcat').change(this, this.toggleCategory);
        // input previous comment into new comment field
        $('.usercomments').change(this, this.inputComment);
    }
    
    // initialize the object
    Comment.prototype.init = function(ifu) {
        this.ifu = ifu;
        this.cubepk = null;
        this.plateid = null;
        this.version = null;
        this.selectedCategory = null;
        this.fxn = 'grabComments';

        this.setParams();
        this.tagname = '#tagfield';
        this.tagbox = utils.initTags(this.tagname);
        this.tags = null;
        this.issueids = null;
        this.comments = null;
    };
    
    // test print
    Comment.prototype.print = function() {
        console.log('We are now printing Comments for ifu: ',this.ifu, this.cubepk, this.version, this.plateid, this.selectedCategory, this.catid);
    };

    // Set the default basic parameters
    Comment.prototype.setParams = function() {
        this.cubepk = parseInt($('#cubepk').val());
        this.plateid = $('#plate').val();
        this.version = $('#drpver').val();
        this.selectedCategory = $('#commentcat option:selected').val();
        this.catid = $('#commentcat option:selected').attr('id');         
    };

    // Hide Issue and Category elements
    Comment.prototype.hideCatsAndIssues = function() {
        $('.catissues').hide();
        $('.catcomments').hide();        
    };

    // Load Issue and Category elements
    Comment.prototype.loadCatsAndIssues = function(value) {
        var val = (value === undefined) ? this.selectedCategory : value;
        $('#catcomment_'+val).fadeIn();
        $('#catissue_'+val).fadeIn();
    };

    // Toggle the comment categories
    Comment.prototype.toggleCategory = function(event) {
        var _this = event.data;
        _this.hideCatsAndIssues();
        // display new things
        _this.selectedCategory = $(this).val();
        _this.catid = $('option:selected',this).attr('id')
        _this.loadCatsAndIssues();
        if (_this.selectedCategory === 'dapqa') {$('.dapqacomms').show();}
    };

    // Reset the comment form
    Comment.prototype.resetForm = function() {
        $('#commentform').modal('hide');
        $('#addcomment_form').trigger("reset");
        $('#commentmessage').empty();
        this.hideCatsAndIssues();
        this.loadCatsAndIssues('general');
        utils.resetTags(this.tagname);
    };

    // Load any Comment texts and checked issues 
    Comment.prototype.loadCommentTextAndIssues = function(id, comment) {
        try {
            //var comment = allcomments[catid];
            var comment_comment = comment.comment;
            var comment_issues = comment.issues;
            var comment_modified = comment.modified;
            if (!!comment_modified) {
                comment_modified = "Submitted on " + comment_modified;
            }
        } catch(err) {
            console.error(err);
            var comment_comment = '';
            var comment_issues = [];
            var comment_modified = '';
            //var comment_modified = err;
        }

        $('#commentfield_'+id).html(comment_comment);
        $('#commentsubmitted_'+id).html(comment_modified);
        try {
            $.each(comment_issues, function(i,value) {
                $('#issue'+value).prop('selected',true);
            });
        } catch(err) {
            $('#commentsubmitted_'+id).html(err);
        }
        $('.selectpicker').selectpicker('refresh');
    };

    // Load Username for comments
    Comment.prototype.loadUsername = function(id, name) {
        var userlabel = $('#userlabel_'+id).text();
        if (userlabel.search(name) < 0) {
            userlabel += name;
        }
        $('#userlabel_'+id).text(userlabel);
    };

    // Load Recent Comments
    Comment.prototype.loadRecentComments = function(id, recentcomments) {
        var htmlstr="<option value='comment0' value='' selected></option>";
        
        $.each(recentcomments, function(i,value) {
            htmlstr += '<option value="comment'+(i+1)+'">'+value+'</option>';
        });
        $('#recentcomments_'+id).html(htmlstr);
    };

    // Populate Comments after grabbing from database
    Comment.prototype.populateComments = function(data) {
        var _this = this;
        $('.catcomments').each(function(index){

            var catid = index+1;

            // load comment text and any checked issues            
            if (data.result.comments !== undefined) {
                _this.loadCommentTextAndIssues(catid,data.result.comments[catid]);
            }

            // load recent comments
            _this.loadRecentComments(catid, data.result.recentcomments[catid]);
            
            // load username 
            _this.loadUsername(catid, data.result.membername);

            // set the new comment text into the object comments
            _this.retrieveComments();
            _this.getIssues('.issuesp');

        });
    };

    // Populate Tags after grabbing from database
    Comment.prototype.populateTags = function(data) {
        var _this = this;
        this.tagbox.suggestions = data.result.alltags;
        
        //if none then returns empty list with length = 0
        if (data.result.tags.length > 0) {
            jQuery.each(data.result.tags, function(i,tag) {
                _this.tagbox.addTag(tag);
            });
        }

        // set the new tags into the object tags
        this.getTags();
    };

    // Input previous comment into new comment field
    Comment.prototype.inputComment = function(event) {
        var _this = event.data;
        var selectedcomment = $('#recentcomments_'+_this.catid+' option:selected').text();
        $('#commentfield_'+_this.catid).text(selectedcomment);
    };

    // Grab comments after successfull login 
    Comment.prototype.grabComments = function grabComments() {
        utils.resetLogin();
        this.getComment();
    };

    // Check the input values into getComment
    Comment.prototype.checkInputs = function(cubepk, ifu) {
        var status = true;

        // check if values are the same
        if (cubepk === ifu) { 
            status = false;
            throw new Error('Error with inputs to getComment. cubepk '+cubepk+' and ifu '+ifu+' are the same');
        }

        /*// check data types for string, cast to int
        $.each({'cubepk':cubepk,'ifu':ifu}, function(key, value) {
            if (typeof value !== 'string') {
                status = false;
                throw new Error('Error with inputs to getComment. Parameter '+key+': '+value+' is not a string value');
            }
        });*/

        // check for numeric input 
        $.each({'cubepk':cubepk,'ifu':ifu}, function(key, value) {
            if ($.isNumeric(value) == false) {
                status = false;
                throw new Error('Error with inputs to getComment. Parameter '+key+': '+value+' is not a numeric value');
            }
        });

        return status;
    };

    // Get comments from database or else log in first
    Comment.prototype.getComment  = function(cubepk, ifu) {
        var cubepk  = (cubepk === undefined) ? this.cubepk : cubepk;
        var ifu  = (ifu === undefined) ? this.ifu : ifu;

        // validate the inputs to getComment
        try {
            var status = this.checkInputs(cubepk,ifu);
        } catch (error) {
            status = false;
            console.log('Error in getComment: '+error);
            Raven.captureException('Error in getComment: '+error);
        }

        // use object values if inputs are no good
        if (status === false) {
            cubepk = this.cubepk;
            ifu = this.ifu;
        }

        var _this = this;
        // set cube pk and ifu form info
        $('#cubepk').val(cubepk);
        $('#ifuname').val(ifu);
                                                        
        $.post($SCRIPT_ROOT + '/marvin/getcomment', {'cubepk':cubepk,'plateid':this.plateid,'ifuname':ifu,'version':this.version},'json') 
            .done(function(data){
                // show commentform if ready (login **may be** necessary)
                if (data.result.ready) {
                    $('#commentform').modal('show');
                    _this.populateComments(data);
                    _this.populateTags(data);
                } else {
                    $('#loginform').modal('show');
                }
            })
            .fail(function(){
                alert('Failed to retrieve comments');
            });   
    };

    // Validate issues
    Comment.prototype.validateIssues = function(issueids) {
        // issueids is not an array
        if (typeof issueids !== Array && typeof issueids !== 'object') {
            throw new Error('Error validating issues: '+issueids+' is not an array');
        }
        // any issueid is not a number
        $.each(issueids, function(index,value) {
            if ($.isNumeric(value) == false) {
                throw new Error('Error validating issues: issue id '+value+' is not a number');
            }
        })

    };

    // Get checked Issues across all categories
    Comment.prototype.getIssues = function(name) {
        var issueids = [];
        var issuelist = utils.getSelected(name);
        if (typeof issuelist === Array || typeof issuelist === 'object') {
            $.each(issuelist, function (index,value) {
                try {
                    issueids.push(parseInt(value.split('issue')[1]));
                } catch (error) {
                    console.error('Error in getComments: can"t split array on issue value properly, '+error);
                    Raven.captureException('Error in getComments: can"t split array on issue value properly, '+error);
                }
            });
        }

        // try issue id validation
        try {
            this.validateIssues(issueids);
        } catch (error) {
            issueids = [];
            console.error('Error in submitComments: '+error);
            Raven.captureException('Error in submitComments: '+error);
        }

        this.issueids = JSON.stringify(issueids);

    };

    // Validate tags
    Comment.prototype.validateTags = function(tags) {
        // tags is not an array
        if (typeof tags !== Array && typeof tags !== 'object') {
            throw new Error('Error validating tags: '+tags+' is not an array');
        }
    };

    // Get tags from comment box
    Comment.prototype.getTags = function() {
        //var tagbox = $(this.tagname).tags();
        var tags = this.tagbox.getTags();
        
        // try tag validation
        try {
            this.validateTags(tags);
        } catch (error) {
            console.error('Error in submitComments: '+error);
            Raven.captureException('Error in submitComments: '+error);
        }
        this.tags = JSON.stringify(tags);
    };

    // Validate comments
    Comment.prototype.validateComments = function(comments) {
        // comments is not an array
        if (typeof comments !== Array && typeof comments !== 'object') {
            throw new Error('Error validating comments: '+comments+' is not an array');
        }
        // comments does not count 4 elements
        if (comments.length !== 4) {
            throw new Error('Error validating comments: '+comments+' does not contain 4 elements');
        }
    };

    // Get comments from all boxes
    Comment.prototype.retrieveComments = function() {
        var comments = [];
        $('.commentfield').each(function(index){
            comments.push(this.value);
        });

        // try validation
        try {
            this.validateComments(comments);
        } catch (error) {
            console.error('Error in submitComments: '+error);
            Raven.captureException('Error in submitComments: '+error);
        }

        this.comments = JSON.stringify(comments);        

    };

    // Submit comments to database
    Comment.prototype.submitComment = function() {
        
        // set object instance
        var _this = this;

        // grab comments from all boxes
        this.retrieveComments();

        // grab all checked issues
        this.getIssues('.issuesp');

        //grab tags
        this.getTags();

        // build form
        var form = {'cubepk':this.cubepk,'plateid':this.plateid,'ifuname':this.ifu,'version':this.version,
            'comments':this.comments,'issueids':this.issueids, 'tags':this.tags};

        $.post($SCRIPT_ROOT + '/marvin/addcomment', form,'json')
            .done(function(data){
                if (data.result.status < 0) {
                    // bad submit
                    _this.resetForm();
                } else {
                    // good response from Inspection, comment ignored with attention message
                    if (data.result.message !== ''){
                        var stat = (data.result.status === 0) ? 'danger' : 'success'; 
                        var htmlstr = "<div class='alert alert-"+stat+"' role='alert'><h4>" + data.result.message + "</h4></div>";
                        $('#commentmessage').html(htmlstr);
                    }
                    // good response from Inspection, comment submitted with success
                    if (data.result.status === 1){
                        setTimeout(Raven.wrap($.proxy(_this.populateComments,_this)), 1500, data);
                        setTimeout(Raven.wrap($.proxy(_this.resetForm,_this)), 1500);
                    }
                }
            })
            .fail(function(){
                alert("Error in response from Inspection webapp.  Please contact admin@sdss.org");
            });   
    };

    return Comment;

})();


