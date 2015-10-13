
QUnit.module('IFU Module', {
    beforeEach: function() {
        this.ifuid = 9101
        this.cubetags = ['test', 'hello', 'this is a new tag'];
    },
    afterEach: function() {
    }
});

QUnit.test('IFU.ifuid_match', 2, function(assert) {
    var ifu = new Ifu(this.ifuid);
    assert.deepEqual(ifu.ifu,this.ifuid,'IFU id matches input');
    assert.deepEqual(ifu.ifuhash,'#'+this.ifuid,'IFU hash matches input');
});

QUnit.test('IFU.set_comments_success', 1, function(assert) {
    var ifu = new Ifu(this.ifuid);
    var comments = new Comment(this.ifuid);
    assert.deepEqual(ifu.comments, comments, 'IFU comments object should be enabled');
});

QUnit.test('IFU.set_dapqa_success', 1, function(assert) {
    var ifu = new Ifu(this.ifuid);
    var dapqa = new Dapqa(this.ifuid);
    assert.deepEqual(ifu.dapqa, dapqa, 'IFU dapqa object should be enabled');
});

QUnit.test('IFU.set_login_function_success', 2, function(assert) {
    var ifu = new Ifu(this.ifuid);
    var loginfxn = 'grabComments';
    assert.deepEqual(ifu.fxn, null, 'IFU login fxn should be null here');
    var fake_event = $.Event('click');
    fake_event.data = ifu;
    ifu.setLoginFxn(fake_event);
    assert.deepEqual(ifu.fxn, loginfxn, 'IFU login fxn should be equal to '+loginfxn);
});

QUnit.test('IFU.set_cubetags_success', 2, function(assert) {
    var ifu = new Ifu(this.ifuid);
    assert.deepEqual(ifu.cubetags, null, 'IFU cube tags should be null');
    ifu.setCubeTags(this.cubetags);
    assert.deepEqual(ifu.cubetags, this.cubetags, 'IFU cube tags should be '+this.cubetags);
});

QUnit.test('IFU.set_cubetags_fail', 2, function(assert) {
    var ifu = new Ifu(this.ifuid);
    assert.deepEqual(ifu.cubetags, null, 'IFU cube tags should be null');
    ifu.setCubeTags();
    assert.deepEqual(ifu.cubetags, null, 'IFU cube tags should still be null');
});

QUnit.test('IFU.set_cubetags_badtype', 2, function(assert) {
    var ifu = new Ifu(this.ifuid);
    var cubetags = 'badtags';
    assert.deepEqual(ifu.cubetags, null, 'IFU cube tags should be null');
    ifu.setCubeTags(cubetags);
    assert.deepEqual(ifu.cubetags, null, 'IFU cube tags should still be null');
});

