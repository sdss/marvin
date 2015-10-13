

var Ifu,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Ifu = (function () {

    function Ifu(ifu) {

        // in case constructor called without new
        if (false === (this instanceof Ifu)) {
            return new Ifu();
        }
        
        this.init(ifu);
        this.setComments();
        
        // Event Handlers
        // set login
        $('#addcommentbut').on('click',this, this.setLoginFxn);
    }
    
    // initialize the object
    Ifu.prototype.init = function(ifu) {
        this.ifu = ifu;
        this.ifuhash = '#'+this.ifu;
        this.cubetags = null;
        this.fxn = null;
        this.comments = null;
        this.dapqa = null;
    };
    
    // test print
    Ifu.prototype.print = function() {
        console.log('We are now printing ifu ',this.ifu, this.cubetags);
    };
    
    // Set the DRP/DAP comment objects for this IFU
    Ifu.prototype.setComments = function() {
        // load Comments object
        try {
            this.comments = new Comment(this.ifu);
        } catch (error) {
            Raven.captureException(error);
            console.error('Error loading Comments:',error);
        }
        // load Dapqa object
        try {
            this.dapqa = new Dapqa(this.ifu);
        } catch (error) {
            Raven.captureException(error);
            console.error('Error loading Dapqa:',error);
        }
    };

    // render the tags for a given IFU
    Ifu.prototype.renderTags = function() {
        var tagdiv = $('#cubetags_'+this.ifu);
        if (tagdiv.length > 0) {
            var ifutags = (this.cubetags === null) ? [] : this.cubetags;
            var tagbox = tagdiv.tags({
                tagData: ifutags,
                tagSize: 'sm',
                readOnly: true
            });
            tagbox.adjustInputPosition();
        }        
    };

    Ifu.prototype.setCubeTags = function(cubetags) {

        // test if undefined
        if (cubetags === undefined ) {
            console.log('Cubetags '+cubetags+' is undefined for IFU '+this.ifu); 
            this.cubetags = null;
            return;
        }

        // test if not an array
        if ($.isArray(cubetags) == false) {
            console.error('Cubetags '+cubetags+' is not a valid array for IFU '+this.ifu);
            this.cubetags = null;
            return;
        }

        // try to set the variable
        try {
            this.cubetags = cubetags;
        } catch (error) {
            Raven.captureException(error);
            this.cubetags = null;
        }
    };

    Ifu.prototype.showCubeTags = function(cubetags) {
        this.setCubeTags(cubetags);
        this.renderTags();
    };

    Ifu.prototype.setLoginFxn = function(event) {
        var _this = event.data;
        _this.fxn = 'grabComments';
        $('#fxn').val(_this.fxn);
        utils.setFunction(_this.comments.grabComments);
    };

    // scroll to the IFU div
    Ifu.prototype.scrollTo = function(location) {
        if (location == 'top') {
            $('html,body').animate({scrollTop:$(this.ifuhash).offset().top},500);
        } else if (location = 'comments') {
            $('html,body').animate({scrollTop:$('#commenttable_'+this.ifu).offset().top},500);
        } else {
            $('html,body').animate({scrollTop:0},500);
        }
    };

    return Ifu;

})();


