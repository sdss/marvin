
QUnit.module('Comments Module', {
    beforeEach: function() {

        this.ifuid = 9101;
        this.cubepk = 10;
        this.plate = '7443';
        this.version = 'v1_3_3';
        this.catid = '1';
        this.selectedcategory = 'general';
        this.tagname = '#tagfield';

        // append divs
        var cubepkdiv = "<input class='hidden' id='cubepk' name='cubepk' value='"+this.cubepk+"'/>"; 
        this.qunitdiv = $('#qunit-fixture');
        this.qunitdiv.append(cubepkdiv);

        // load IFU + comments
        this.ifu = new Ifu(this.ifuid);
        this.ifu.comments.issueids = null;
        this.cubetags = ['test', 'hello', 'this is a new tag'];
        ifu = this.ifu;

    },
    afterEach: function() {
        // reset things
        this.ifu.comments.resetForm();
        //$('#addcomment_form').trigger("reset");
        this.ifu = null;
    }
});

QUnit.test('Comments.comments_loaded',1,function(assert) {
    assert.ok(this.ifu.comments, 'The comments object should be loaded');
});

QUnit.test('Comments.setParams_success', 6, function(assert) {

    assert.deepEqual(this.ifu.comments.ifu, this.ifuid, 'IFU id should be set in comments to '+this.ifuid);
    assert.deepEqual(this.ifu.comments.plateid, this.plate, 'Plate id should be set in comments to '+this.plate);
    assert.deepEqual(this.ifu.comments.cubepk, this.cubepk, 'cube pk should be set in comments to '+this.cubepk);
    assert.deepEqual(this.ifu.comments.version, this.version, 'version should be set in comments to '+this.version);
    assert.deepEqual(this.ifu.comments.catid, this.catid, 'category id should be set in comments to '+this.catid);
    assert.deepEqual(this.ifu.comments.selectedCategory, this.selectedcategory, 'selected category should be set in comments to '+this.selectedcategory);

});

QUnit.test('Comments.setParams_tagbox_initialized', 3, function(assert) {
    assert.deepEqual(this.ifu.comments.tagname,this.tagname,'Tagname should be set in comments to '+this.tagname);
    assert.ok(this.ifu.comments.tagbox,'comments tag box should be initialized');
    assert.deepEqual(this.ifu.comments.tagbox.tagData,[],'comments tag data should be equal to empty array');
});


QUnit.test('Comments.checkInputs_success',1, function(assert) {

    function checkInputs(cube,ifu,expected, expected_error, msg) {
        var expected_status = expected;
        var status = this.ifu.comments.checkInputs(cube,ifu);
        assert.deepEqual(status,expected_status, msg);
    }

    checkInputs(this.cubepk, this.ifuid, true, 'Values are good');

});


QUnit.test('Comments.checkInputs_failed',6, function(assert) {

    function checkInputs(cube,ifu,expected, expected_error, msg) {
        var expected_status = expected;
        try {
            var status = this.ifu.comments.checkInputs(cube,ifu);
        } catch (error) {
            status = false;
            assert.deepEqual(error,expected_error, msg);
            assert.deepEqual(status,expected_status, 'Status should be '+expected_status);
        }
    }

    checkInputs(this.ifuid, this.ifuid, false, Error('Error with inputs to getComment. cubepk '+this.ifuid+' and ifu '+this.ifuid+' are the same'));
    checkInputs('54a', this.ifuid, false, Error('Error with inputs to getComment. Parameter cubepk: 54a is not a numeric value'));
    checkInputs(this.cubepk, '9101a', false, Error('Error with inputs to getComment. Parameter ifu: 9101a is not a numeric value'));
    //checkInputs(54, this.ifuid, false, Error('Error with inputs to getComment. Parameter cubepk: 54 is not a string value'));
    //checkInputs(this.cubepk, 9101, false, Error('Error with inputs to getComment. Parameter ifu: 9101 is not a string value'));
});

QUnit.test('Comments.getIssues_success', 8, function(assert) {

    var name = '.issuesp';

    function checkIssues(expected, msg) {
        assert.notOk(this.ifu.comments.issuesids,'initial issueids should be null');
        this.ifu.comments.getIssues(name);
        assert.deepEqual(this.ifu.comments.issueids, expected, msg);
        this.ifu.comments.issuesids = null; //reset
        this.ifu.comments.resetForm();
        $(name).selectpicker('deselectAll');
        $(name).selectpicker('refresh');        
    }

    // empty issues
    checkIssues('[]','Issue ids should be an empty array');
    // selected issues
    $(name).selectpicker('val','issue1');
    checkIssues('[1]','Issue id 1, wrong redshift should be selected');
    $(name).selectpicker('val',['issue1','issue2']);
    checkIssues('[1,2]','Issue id 1 and 2, wrong redshift,wrong effective radius should be selected');
    $(name).selectpicker('val',['issue4','issue5', 'issue9','issue12']);
    checkIssues('[4,5,9,12]','Issue id 4,5,9,12 should be selected');
});

QUnit.test('Comments.getIssues_failure', 4, function(assert) {

    function validate(issues, expected, msg) {
        var name = '.issuesp';
        assert.notOk(this.ifu.comments.issuesids,'initial issueids should be null');
        try {
            this.ifu.comments.validateIssues(issues);
        } catch (error) {
            assert.deepEqual(error, expected, msg);
        }
        this.ifu.comments.issuesids = null; //reset
        this.ifu.comments.resetForm();
    }

    validate('issue1', Error('Error validating issues: issue1 is not an array'),'Error should be caught on issuesids not being an array');
    validate(['1','2a'],Error('Error validating issues: issue id 2a is not a number'), 'Error should be caught on any issuesids not being a valid number')

});

QUnit.test('Comments.getTags_success', 6, function(assert) {

    function checkTags(expected, msg) {
        assert.notOk(this.ifu.comments.tags,'initial tags should be null');
        this.ifu.comments.getTags();
        assert.deepEqual(this.ifu.comments.tags, expected, msg);
        this.ifu.comments.tags = null; //reset
        utils.resetTags(this.ifu.comments.tagname);
    }

    // add none
    checkTags('[]','Tags should be an empty array');
    // add one
    this.ifu.comments.tagbox.addTag('hello');
    checkTags('["hello"]','New tags should say hello');
    // add a bunch
    var _this = this;
    var tmptags = ["this","is","a","new","tag"];
    $.each(tmptags, function(index,tag) {
        _this.ifu.comments.tagbox.addTag(tag);
    })
    checkTags(JSON.stringify(tmptags),'New tags should say '+JSON.stringify(tmptags));

});

QUnit.test('Comments.getTags_failure', 2, function(assert) {

    function validate(tags, expected, msg) {
        assert.notOk(this.ifu.comments.tags,'initial issueids should be null');
        try {
            this.ifu.comments.validateTags(tags);
        } catch (error) {
            assert.deepEqual(error, expected, msg);
        }
        this.ifu.comments.tags = null; //reset
        utils.resetTags(this.ifu.comments.tagname);
    }

    validate('tag1', Error('Error validating tags: tag1 is not an array'),'Error should be caught on tags not being an array');

});

QUnit.test('Comments.retrieveComments_success', 6, function(assert) {

    function insertComments(comments) {
        $.each(comments, function(index,value) {
            $('#commentfield_'+(index+1)).html(value);
        });
    }

    function checkComments(expected, msg) {
        assert.notOk(this.ifu.comments.comments,'initial comments should be null');
        this.ifu.comments.retrieveComments();
        assert.deepEqual(this.ifu.comments.comments, expected, msg);
        var tmparr = JSON.parse(this.ifu.comments.comments);
        assert.deepEqual(tmparr.length, 4, 'Array length should be 4');
        this.ifu.comments.comments = null; //reset
        $('#addcomment_form').trigger("reset");
    }

    // add empty
    checkComments('["","","",""]','Comments should be an empty array of 4 elements');
    // add some
    insertComments(["hello","bad redshift","","testing"]);
    checkComments('["hello","bad redshift","","testing"]','Comments should be a filled array of 4 elements');

});

QUnit.test('Comments.retrieveComments_failure', 4, function(assert) {

    function validate(comments, expected, msg) {
        assert.notOk(this.ifu.comments.comments,'initial comments should be null');
        try {
            this.ifu.comments.validateComments(comments);
        } catch (error) {
            assert.deepEqual(error, expected, msg);
        }
        this.ifu.comments.comments = null; //reset
        $('#addcomment_form').trigger("reset");
    }

    validate('comment1', Error('Error validating comments: comment1 is not an array'),'Error should be caught on comments not being an array');
    validate(['hello','test'], Error('Error validating comments: hello,test does not contain 4 elements'),'Error should be caught on comments not having 4 elements');

});

QUnit.test('Comments.populateTags_success', 6, function(assert) {
    var alltags = ['hello','world','galaxy','help','test'];
    var tags = ['hello','new','test','tag'];
    var data = {'result':{'alltags':alltags, 'tags':tags, 'ready':true}};
    var tagdiv = $('#tagfield span');

    assert.notOk(this.ifu.comments.tags,'initial tags should be null');
    assert.deepEqual(tagdiv.length,0,'initial tag field should have no spans');
    this.ifu.comments.populateTags(data);
    // check object tags
    assert.deepEqual(this.ifu.comments.tags, JSON.stringify(tags), 'Object tags should be set to new tags '+tags);
    assert.deepEqual(this.ifu.comments.tagbox.suggestions, alltags, 'Suggested tags in tagbox should be set to alltags');
    // check actual div elements
    var tagdiv = $('#tagfield span');
    assert.deepEqual(tagdiv.length,tags.length,'tag field should have spans of count '+tags.length);
    var newtags = [];
    tagdiv.map(function() {
        newtags.push($(this).html());
    });
    assert.deepEqual(newtags, tags, 'names of tags in field should be same as tags '+tags);    

});

QUnit.test('Comments.loadUsername_success', 2, function(assert) {

    var id = 1;
    var origtext = "Previous Comments on General from User: ";
    var text = $('#userlabel_'+id).text();
    // initial
    assert.deepEqual(text, origtext, 'initial text has no name');

    // add name and re-check
    var name = 'Brian Cherinka';
    this.ifu.comments.loadUsername(id,name);
    var text = $('#userlabel_'+id).text();
    assert.deepEqual(text, origtext+name, 'new text has a name');

    // reset
    $('#userlabel_'+id).text(origtext);
    $('#addcomment_form').trigger("reset");
});


QUnit.test('Comments.loadRecentComments_success', 4, function(assert) {

    var id = 2;
    var recentdiv = $('#recentcomments_'+id);
    var text = $('#recentcomments_'+id).html();
    assert.deepEqual(recentdiv.find('option').length,0,'initial recent select should be empty');    
    assert.ok(!text.trim(),'initial recent comments should be empty');

    // add recent comment
    var recent = ['This is a my prior comment.'];
    this.ifu.comments.loadRecentComments(id,recent);
    var recentdiv = $('#recentcomments_'+id);
    var text = $('#recentcomments_'+id).html();
    assert.deepEqual(recentdiv.find('option').length,2,'recent select should now have 2 elements');    
    assert.ok(text.trim(),'recent comments should be: '+recent);

    // reset
    $('#addcomment_form').trigger("reset");
    recentdiv.find('option').remove();

});

QUnit.test('Comments.loadCommentTextAndIssues_success', 7, function(assert) {

    var id = 3;
    var comment = {'comment':'here is a new comment', 'issues':[5,7,9], 'modified':'time is now'};
    var commtext = $('#commentfield_'+id).html();
    var commmod = $('#commentsubmitted_'+id).html();
    var commiss = $('.issuesp :selected');
    assert.ok(!commtext.trim(),'initial comment field text should be empty');
    assert.ok(!commmod.trim(),'initial comment modified text should be empty');
    assert.deepEqual(commiss.length,0,'initial comment issues should be 0');

    // insert new comment
    this.ifu.comments.loadCommentTextAndIssues(id,comment);
    var commtext = $('#commentfield_'+id).html();
    var commmod = $('#commentsubmitted_'+id).html();
    var commiss = $('.issuesp :selected');
    assert.deepEqual(commtext,comment.comment,'new comment text should be '+comment.comment);
    assert.deepEqual(commmod,'Submitted on '+comment.modified,'new comment modified time should be '+comment.modified);
    assert.deepEqual(commiss.length,comment.issues.length,'new comment issues should be '+comment.issues.length);

    var newiss = [];
    commiss.map(function() {
        newiss.push(parseInt($(this).attr('id').split('issue')[1]));
    });
    assert.deepEqual(newiss, comment.issues, 'newly selected issue ids should be same as '+comment.issues);    

});

QUnit.test('Comments.populateComments_success',4,function(assert) {

    var recent = {'1':[""],'2':["This is a prior comment"],'3':[""],'4':[""]};
    var data = {'result': {'membername':'Brian Cherinka', 'recentcomments': recent, 
    'comments':{'1':{},'2':{},'3':{'comment':'here is a new comment', 'issues':[5,7,9], 'modified':'time is now'},'4':{}} }};

    assert.notOk(this.ifu.comments.comments,'initial comments is null');
    assert.notOk(this.ifu.comments.issuesids,'initial issuesids is null');

    this.ifu.comments.populateComments(data);

    var issueids = JSON.stringify([5,7,9]);
    var comms = JSON.stringify(["","","here is a new comment",""]);

    assert.deepEqual(this.ifu.comments.comments,comms, 'new comments should be '+comms);
    assert.deepEqual(this.ifu.comments.issueids,issueids,'new issuesids should be '+issueids);


});



