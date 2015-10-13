

var Plateinfo,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Plateinfo = (function () {

    function Plateinfo(plateid,imagediv, allcubetags) {

        // in case constructor called without new
        if (false === (this instanceof Plateinfo)) {
            return new Plateinfo();
        }

        this.init(plateid, imagediv, allcubetags);

        // Event Handlers
        
        // toggle plate design
        $('#pdtoggle').on('click', this, this.togglePlateDesign);
        // show IFU when clicked
        //this.ifuimages.on('click', $.proxy(this.displayIFU, this));
        // shift ifu panes on keyup
        $(window).on('keyup', $.proxy(this.switchIFUs,this));
    }
    
    // initialize the object
    Plateinfo.prototype.init = function(plateid, imagediv, allcubetags) {
        this.plateid = (plateid == undefined) ? null : plateid;
        this.hash = null;
        this.ifuhash = null;
        this.selectedifu = null;
        this.currentifu = null;
        this.commentinhash = false;
        this.mainimagediv = null;
        this.ifuimages = null;
        this.ifus = {};
        this.allcubetags = allcubetags;

        // plate design info
        this.platedata = null;
        this.platera = null;
        this.platedec = null;

        //methods
        //this.getIFUHash();
        //this.setIFUs(imagediv);
        //this.showIFU();
    };
    
    // test print
    Plateinfo.prototype.print = function() {
        console.log('We are now printing plateinfo on',this.plateid,this.selectedifu, this.ifus[this.selectedifu]);
    };
    
    // get IFU hash
    Plateinfo.prototype.getIFUHash = function() {
        if (this.hash.search('comment') >= 0) {
            this.ifuhash = this.hash.slice(this.hash.search('_')).replace('_','#');
            this.commentinhash = true;
        } else if (this.hash) {
            this.ifuhash = this.hash;
        }

        this.selectedifu = (this.ifuhash) ? this.ifuhash.slice(this.ifuhash.search('#')+1) : null;
        return this.ifuhash;     
    };

    // set the div for the plate ifu images 
    Plateinfo.prototype.setIFUs = function(imdiv) {
        var _this = this;
        this.mainimagediv = imdiv;
        this.ifuimages = this.mainimagediv.find('.ifuims');

        // build a list of IFU objects for the given plate
        $.each(this.ifuimages,function() {
            var ifuid = this.hash.slice(this.hash.search('#')+1);
            _this.ifus[ifuid] = new Ifu(ifuid);
            _this.ifus[ifuid].setCubeTags(_this.allcubetags[ifuid]);
        });

    };

    // load an IFU
    Plateinfo.prototype.loadIfu = function(ifuid) {
        if (ifuid !== 'None') {
            this.currentifu = new Ifu(ifuid);
            this.selectedifu = ifuid;
            this.currentifu.setCubeTags(this.allcubetags[ifuid]);
            this.currentifu.renderTags();
            ifu = this.currentifu;
        }
    };

    // toggle the plate design D3 window
    Plateinfo.prototype.togglePlateDesign = function(event) {
        var _this = event.data;
        if ($(this).hasClass('active')){
            $('#platedesign').hide();
            $('#platedesign').empty();
        } else {
            $('#platedesign').show();
            $('#platedesign').html('<h3><small>Unfinished</small></h3>');
            loadPlateDesign();
        }
    };

    // event handler for showing IFU on click
    Plateinfo.prototype.displayIFU = function(event) {
        this.ifuhash = $(event.currentTarget).attr('href');
        this.selectedifu = this.ifuhash.slice(this.ifuhash.search('#')+1);
        $('.galinfo').hide();
        this.showIFU();
        this.currentifu = this.ifus[this.selectedifu];
        // set global ifu
        ifu = this.currentifu;
    };

    // display ifu div element
    Plateinfo.prototype.showIFU = function() {
        if (this.ifuhash) {
            $(this.ifuhash).fadeIn();  
            if (this.selectedifu) this.ifus[this.selectedifu].renderTags();
        }
        if (this.commentinhash) $(this.hash).fadeIn();   
        this.currentifu = this.ifus[this.selectedifu];

        // set global ifu
        ifu = this.currentifu;     
    };

    // switch IFU panes on key presses
    Plateinfo.prototype.switchIFUs = function(event) {
        if (this.ifuhash) {
            var gals = $('.cubedetails');
            var isShift = event.shiftKey;
            if (isShift) {
                $('.galinfo').hide();
                switch(event.keyCode) {
                    case 37:
                        this.pan(gals, 'left');
                        break;
                    case 39:
                        this.pan(gals, 'right');
                        break;
                }
            }
        }
    };

    // pan the IFU panes
    Plateinfo.prototype.pan = function(gals, dir) {
        if (dir == 'left') {
            this.selectedifu = gals.find(this.ifuhash).prev('.galinfo').attr('id');
        } else if (dir == 'right') {
            this.selectedifu = gals.find(this.ifuhash).next('.galinfo').attr('id');
        }
        this.ifuhash = '#'+this.selectedifu;
        this.showIFU();
        this.selectedfu = (typeof this.selectedifu !== 'undefined') ? this.selectedifu : null;
        window.location.hash = (this.selectedifu) ? this.ifuhash : '';
    }

    return Plateinfo;

})();


