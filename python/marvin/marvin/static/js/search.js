

var Search,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Search = (function () {

    function Search(nsamagids) {

        // in case constructor called without new
        if (false === (this instanceof Search)) {
            return new Search();
        }
        
        this.init(nsamagids);
        
        // Event Handlers
        $('#searchon').on('change',this,this.showMode);
        $('#searchform').on('submit', this, this.buildSearchForm);
        $('#commentform').on('submit', this, this.buildCommentForm);
        $('.nav-tabs').on('click','li a', this, this.clearResults);
        $('#datetoggle').on('click', this, this.toggleDateMode);
        $('#radecmode a').on('click', this, this.toggleRADecMode);
        $('#nsaband a').on('click', this, this.showNsaMags);
        $('#nsainfo a').on('click', this, this.showNsaBands);
    }
    
    // initialize the object
    Search.prototype.init = function(nsamagids) {
        this.nsamagids = nsamagids;
        this.searchform = null;
        this.commentform = null;
        this.sqlform = null;
    };
    
    // test print
    Search.prototype.print = function() {
        console.log('We are now printing search info: ');
    };

    // Show search display options
    Search.prototype.showMode = function(event) {
        $('.search').hide();
        $('#by'+$(this).val()).fadeIn();
    };

    // Set list for DRP issue  ids
    Search.prototype.setDRPIssueIds = function() {
        var issuelist = utils.getSelected($('#drpissue'));
        $('#issues').val(issuelist);
    }; 

    // Set list for DAP issue  ids
    Search.prototype.setDAPIssueIds = function() {
        var dapissuelist =  utils.getSelected($('#dapissue'));
        $('#dapissues').val(dapissuelist);
    }; 

    // Set list for tag  ids
    Search.prototype.setTagIds = function() {
        var taglist = utils.getSelected($('.tagsp'));
        $('#tagids').val(taglist);
    }; 

    // Set list for default search ids
    Search.prototype.setDefaultIds = function() {
        var defaultlist = utils.getSelected($('.defsp'));
        $('#defaultids').val(defaultlist);
    }; 

    // Make search lists and add them to the form
    Search.prototype.makeLists = function(type) {
        if (type == 'search') {
            this.setDefaultIds();
            this.setTagIds();
        } else if (type == 'comment') {
            this.setDRPIssueIds();
            this.setDAPIssueIds();
        }
    };

    // Build search form
    Search.prototype.buildSearchForm = function(event) {
        var _this = event.data;
        _this.makeLists('search');
        _this.searchform = $('#searchform').serialize();
    };

    // Build comment form
    Search.prototype.buildCommentForm = function(event) {
        var _this = event.data;
        _this.makeLists('comment');
        _this.commentform = $('#commentform').serialize();
    };

    // Get SQL of search form
    Search.prototype.getSQL = function() {
        var event = $.Event('click');
        event.data = this;
        this.buildSearchForm(event);
            
        $.post($SCRIPT_ROOT + '/marvin/getsql/', this.searchform,'json')
            .done(function(data){
                var htmlstr='';
                for (var i=0; i<data.result['rawsql'].length; ++i){
                    htmlstr += '<div class="row"><p>'+data.result['rawsql'][i]+'</p></div>';
                }
                $('#rawsql').html(htmlstr);
                $('#sql').text(data.result['sql']);
            })
            .fail(function(data){
                $('#sql').text('Server request failed to grab sql!');
            });
    };

    // Submit the SAS table form
    Search.prototype.submitTableForm = function() {
        var form = $('#sastableform');
        var data = $('.sastable').bootstrapTable('getData');
        $('#hiddendata').val(JSON.stringify(data));
        form.submit();
        $('#fitsform').modal('hide');        
    }

    // Reset SQL form
    Search.prototype.resetSQL = function() {
        $('#sqlinput').val('');
    };

    // Reset comment form
    Search.prototype.resetComments = function() {
        var _this = this;
        $('#comment_form').trigger("reset");
        var form = $('#comment_form :input[type="text"]');
        form.each(function() {
            $(this).val('');
        });
        $('#commtag option[value=Any]').attr('selected','selected');
        $('#cat option[value=any]').attr('selected','selected');
        $('.selectpicker :checked').each(function() {
            this.selected = false;
        });
        $('.selectpicker').selectpicker('refresh');
    };

    // Clear any results 
    Search.prototype.clearResults = function(event) {
        $('#results').hide();
    };

    // Toggle RA/Dec mode
    Search.prototype.toggleRADecMode = function(event) {
        var id = $(this).attr('id');
        var input = $('#radectext');
        if (id == 'cone') {
            input.prop('disabled',false); 
            input.attr('placeholder','Cone: input radius [deg]');
            $('#radechidden').val(id);
        } else {
            input.prop('disabled',true);
            input.attr('placeholder','Between mode');
            input.val('');
            $('#radechidden').val(id);
        }   
    };

    // Toggle DateTime Option
    Search.prototype.toggleDateMode = function(event) {
        if ($(this).hasClass('active')){
            $(this).button('reset');
            $('#datehide').val('before');
        } else {
            $(this).button('complete');
            $('#datehide').val('after');
        }
    };

    // MAKE THIS BETTER 

    // Show NSA magnitude bands 
    Search.prototype.showNsaMags = function(event) {
        var id = $(this).attr('id');
        var mag = id.slice(7);
        var inputid = $('.nsa :visible').attr('id');
        var finalid = 'by'+inputid.slice(0,inputid.length-1)+mag;
        $('.nsa').hide();
        $('#'+finalid).show();
    };

    // Show NSA info ??
    Search.prototype.showNsaBands = function(event) {
        var _this = event.data;
        var id = $(this).attr('id');
        var number = parseInt(id.slice(3));
        var magarr = _this.nsamagids;
        var href = '#by'+id;
        $('.nsa').hide();
        $('#nsabandbutton').hide();
        
        if ($.inArray(number,magarr) > -1) {
            $('#nsabandbutton').show();
            $(href+'f').show();
        } else {
            $(href).show();
        }
    };

    return Search;

})();


