QUnit.module('Plate Module', {
    beforeEach: function() {
        this.plate = 7443
        this.cubetags = ['test', 'hello', 'this is a new tag'];
    },
    afterEach: function() {
    }
});

QUnit.test('Plate.plateid_match', 1, function(assert) {
    var plate = new Plateinfo(this.plate, $('.plateifu_images'), this.cubetags);
    assert.deepEqual(plate.plateid,this.plate,'Plate id matches input');
});

