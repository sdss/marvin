
QUnit.module('Header Module', {
    beforeEach: function() {
        this.header = new Header('MPL');
        this.savedAjax = this.header.sendAjax;
        this.header.sendAjax = function() {
            console.log('test version of Ajax call');
        }
    },
    afterEach: function() {
        //resetHeader('MPL');
        //this.header.savedAjax = this.savedAjax;
    }
});

function resetHeader(vermode, searchid) {
    this.header.versionmode = vermode;
    changeSearch(searchid);
    this.header.setParams();
    this.header.showVersion();
}

/*
QUnit.test('Header.checkDefaults', 5, function(assert) {
    function checkParams(name, input, expected) {
        assert.deepEqual(input,expected,name+' matches as expected');
    }

    resetHeader('MPL','plateid');
    checkParams('header.versionmode', this.header.versionmode,'MPL');
    checkParams('header.versionid', this.header.versionid,'mpl');
    checkParams('header.searchid', this.header.searchid,'plateid');
    checkParams('header.searchtext', this.header.searchtext,'Plate ID');
    checkParams('header.typetext', this.header.typetext,'MPL');
});

/*
// check if the correct version div element is displayed
QUnit.test('Header.correct_versiondiv_shown', 1, function(assert) {
    var mode = this.header.versionmode;
    assert.ok(this.header.isVisible(this.header.versionid), this.header.versionid+' div is visible');
});

// check if the version text in dropdown-button is same as version select elements
QUnit.test('Header.versiontext_matches_versionmode', 1, function(assert) {
    var vertype = 'mplver';//this.header.getVisibleVersionID();
    var vertext = (vertype == 'mplver') ? 'MPL' : 'DRP/DAP'; 
    var newhtml = 'Set Version By: '+vertext+' <span class="caret"></span>'; 
    var buthtml = $('#verbut').html().trim(); 
    assert.htmlEqual(buthtml,newhtml,'Html in Version button matches version mode displayed');
});

*/

// test the form output
QUnit.test('Header.buildForm', 1, function(assert) {
    var form = this.header.buildForm();
    var expect_form = [{'name':'mplver','value':'MPL-3'},{'name':'version','value':'v1_3_3'},
    {'name':'dapversion','value':'v1_0_0'},{'name':'vermode','value':'mpl'}];

    assert.deepEqual(form,expect_form, 'Header form is in correct shape');

});

// simulate a version change
function changeVersion(mode,id) {
    $('#'+mode).find('option:selected').removeAttr("selected");
    $('#'+mode+' option[id='+id+']').prop('selected',true);
}

// trigger version change
function triggerVersionChange() {
    var fake_event = $.Event('change');
    $('.verselecttype').trigger(fake_event,this,this.header.sendVersionInfo);
}

/*QUnit.test('Header.toggleVersionChange',1, function(assert) {

    changeVersion('mplver','MPL-3');
    assert.equal(true,'fdfdfd');
    //triggerVersionChange();
});*/

// simulate a change in search mode
function changeSearch(id) {
    $("#idselect").find('option:selected').removeAttr("selected");
    $('#idselect option[id='+id+']').prop('selected',true);
}
// trigger toggleSearchType event
function triggerSearchChange() {
    var fake_event = $.Event('change');
    $('#idselect').trigger(fake_event, this, this.header.toggleSearchType);    
}

QUnit.test('Header.toggleSearchType_plateid_to_mangaid', 5, function(assert) {

    function idtext_matches(idexpected, textexpected) {
        var name = $('#idtext').attr('name');
        var text = $('#idtext').attr('placeholder');
        assert.deepEqual(name,idexpected,'Search toggle id matches '+idexpected);
        assert.deepEqual(text,textexpected,'Search toggle text matches '+textexpected);
    }

    function checkParams(name, input, expected) {
        assert.deepEqual(input,expected,name+' matches as expected');
    }

    checkParams('header.searchid',this.header.searchid,'plateid');

    // to MaNGA ID
    changeSearch('mangaid');
    triggerSearchChange();
    idtext_matches('mangaid', 'MaNGA ID');
    checkParams('header.searchid', this.header.searchid,'mangaid');
    checkParams('header.searchtext', this.header.searchtext,'MaNGA ID');
   
});

QUnit.test('Header.toggleSearchType_mangaid_to_plateid', 5, function(assert) {

    function idtext_matches(idexpected, textexpected) {
        var name = $('#idtext').attr('name');
        var text = $('#idtext').attr('placeholder');
        assert.deepEqual(name,idexpected,'Search toggle id matches '+idexpected);
        assert.deepEqual(text,textexpected,'Search toggle text matches '+textexpected);
    }

    function checkParams(name, input, expected) {
        assert.deepEqual(input,expected,name+' matches as expected');
    }

    checkParams('header.searchid',this.header.searchid,'mangaid');

    // to Plate ID
    changeSearch('plateid');
    triggerSearchChange();
    idtext_matches('plateid', 'Plate ID');
    checkParams('header.searchid', this.header.searchid,'plateid');
    checkParams('header.searchtext', this.header.searchtext,'Plate ID');
   
});

