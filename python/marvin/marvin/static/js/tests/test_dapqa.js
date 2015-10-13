QUnit.module('DapQA Module', {
    beforeEach: function() {
        this.ifuid = 9101;
        this.cubepk = 10;
        this.plate = '7443';
        this.version = 'v1_2_0'; 
        this.dapversion = 'v1_0_0';
        this.ready = true;

        // append divs
        var cubepkdiv = "<div id='"+this.ifuid+"'><input class='hidden' id='cubepk' name='cubepk' value='"+this.cubepk+"'/></div>"; 
        var inspectdiv = "<input class='hidden' id='inspectready' value='"+this.ready+"'/>";
        this.qunitdiv = $('#qunit-fixture');
        this.qunitdiv.append(cubepkdiv,inspectdiv);

        this.dapqa = new Dapqa(this.ifuid);
        this.dapqa.setDefault();
        this.dapqa.tags = null;
        this.dapqa.issues = null;

    },
    afterEach: function() {
        this.dapqa = null;
    }
});


QUnit.test('DapQA.general_load', 1, function(assert) {

    console.log(this.dapqa);
    assert.deepEqual(true,true,'fdfd');
});


QUnit.test('Dapqa.grabTags_success', 4, function(assert) {

    var _this = this;

    function insertTags(tags) {
        $.each(tags,function(i,tag) {
            _this.dapqa.tagbox.addTag(tag);
        });        
    }

    function checkTags(input, expected) {
        assert.notOk(_this.dapqa.tags,'initial tags should be null');
        if (input !== null) {insertTags(input);}
        _this.dapqa.grabTags();
        assert.deepEqual(_this.dapqa.tags,expected,'new tags should be set to '+expected);
        _this.dapqa.tags = null;
        utils.resetTags(_this.dapqa.tagname);
    }
    
    // test insert tags
    var tags = ['hello','dapqa tag'];
    checkTags(tags,JSON.stringify(tags));
    // test no tags
    checkTags(null,'[]');    

});


QUnit.test('Dapqa.grabTags_failure', 2, function(assert) {

    var _this = this;
    function validate(tags, expected, msg) {
        assert.notOk(_this.dapqa.tags,'initial tags should be null');
        try {
            _this.dapqa.validateTags(tags);
        } catch (error) {
            assert.deepEqual(error, expected, msg);
        }
        _this.dapqa.tags = null; //reset
        utils.resetTags(_this.dapqa.tagname);
    }

    validate('tag1', Error('Error validating tags: tag1 is not an array'),'Error should be caught on tags not being an array');

});

QUnit.test('Dapqa.parseDapIssues_success', 4, function(assert) {

    var _this = this;
    var name = '#dapqacomment_form_'+this.ifuid+' select[id*="dapqa_issue_'+this.dapqa.key+'"]';

    function checkIssues(input, expected) {

        $(name).selectpicker('deselectAll');
        $(name).selectpicker('refresh'); 

        assert.notOk(_this.dapqa.issues,'initial issues should be null');
        _this.dapqa.parseDapIssues();
        assert.deepEqual(_this.dapqa.issues,expected,'new issues should be set to '+expected);
        _this.dapqa.issues = null;
        _this.dapqa.mainform.trigger('reset');

    }

    // test no issues
    checkIssues(null,'"any"');
    // add issues
    var issues = ['issue_1_1', 'issue_3_1', 'issue_2_4', 'issue_6_4', 'issue_5_6'];
    $(name).selectpicker('val',issues);
    checkIssues(issues,JSON.stringify(issues));

});

QUnit.test('Dapqa.parseDapIssues_failure', 12, function(assert) {

    var _this = this;
    function validate(issues, expected, msg) {
        assert.notOk(_this.dapqa.issues,'initial issues should be null');
        try {
            _this.dapqa.validateIssues(issues);
        } catch (error) {
            assert.deepEqual(error, expected, msg);
        }
        _this.dapqa.issues = null; //reset
        _this.dapqa.mainform.trigger('reset');
    }

    validate('issue1', Error('Error validating issues: issue1 is not an array or any'),'Error should be caught on issues not being an array / "any"');
    validate(['issue_1'], Error('Error validating issues: issue_1 element not splittable; does not have correct format'),'Error should be caught on issue not having correct length');
    validate(['junk_1_5'],Error('Error validating issues: 1st element of junk_1_5 not issue'), 'Error should be caught on 1st element not set to phrase issue');
    validate(['issue_a_4'], Error('Error validating issues: 2nd element of issue_a_4 not a number'), 'Error should be caught on 2nd element not being numeric');
    validate(['issue_3_a'], Error('Error validating issues: 3rd element of issue_3_a not a number or outside range 1-6'), 'Error should be caught on 3rd element not a number or outside range 1-6');
    validate(['issue_3_9'], Error('Error validating issues: 3rd element of issue_3_9 not a number or outside range 1-6'), 'Error should be caught on 3rd element not a number or outside range 1-6');

});

QUnit.test('Dapqa.buildDapForm_success', 1, function(assert) {


    var dapform = this.dapqa.buildDapForm();
    //console.log('test',dapform);
    assert.deepEqual(true,true,'dfd');
});


QUnit.test('Dapqa.buildDapForm_failed', 2, function(assert) {

    var dapform;
    var _this = this;
    function validate(input, expected, msg) {
        try {
            _this.dapqa.validateForm(input)
        } catch (error) {
            assert.deepEqual(error, expected, msg);
        }
        _this.dapqa.mainform.trigger('reset');
        //_this.dapqa.issues = null;
    }

    function changeForm(name, value, msgbit) {
        msg = Error('Error validating form: Parameter '+name+' with value '+value+' '+msgbit);
        $.each(dapform, function(i,param) {
            if (param.name == name) param.value = value;
        });
        return msg;
    }

    function setFormParams() {
        var newdata = [{'name':'key','value':_this.dapqa.key},{'name':'mapid','value':_this.dapqa.mapid},{'name':'cubepk','value':_this.dapqa.cubepk},
                       {'name':'qatype','value':_this.dapqa.qatype},{'name':'issues','value':'any'},
                       {'name':'tags','value': '[]'}]
        return newdata;
    }

    function resetForm() {
        var newdata = setFormParams();
        dapform = _this.dapqa.buildDapForm(newdata);        
    }

    // set initial form
    resetForm();
    // test if a value is not a string
    msg = changeForm('mapid',5,'is not a string');
    validate(dapform, msg, 'Error should be caught on element not a string: mapid 5');

    resetForm();
    msg = changeForm('dapqa_comment1_1',['',''],'is not a string');
    validate(dapform, msg, 'Error should be caught on element not a string: dapqa_comment1_1 [","]');

    
    
});



