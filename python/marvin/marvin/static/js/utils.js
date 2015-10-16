// Javascript code for general things

var Utils,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Utils = (function() {

	// Constructor
    function Utils() {

        // in case constructor called without new
        if (false === (this instanceof Utils)) {
            return new Utils();
        }
        
        this.init();

        // event handlers
        
        // button collapse name change toggle
        $('.butcollapse').on('click',this,this.changeName);
        // login handlers 
        $('#username').on('keyup', this, this.submitLogin);
        $('#password').on('keyup', this, this.submitLogin);
        $('#loginform').on('shown.bs.modal', this, this.focusLogin);
        
    }

    // initialize the object
    Utils.prototype.init = function() {
        this.fxn = null;
    };

    // Grab a GET parameter from the URL
    Utils.prototype.GetURLParameter = function(sParam) {
		var sPageURL = window.location.search.substring(1);
		var sURLVariables = sPageURL.split('&');
		
		for (var i = 0; i < sURLVariables.length; i++){
			 var sParameterName = sURLVariables[i].split('=');
			 if (sParameterName[0] == sParam){
			 	return sParameterName[1];
			 }
		}
    };

    // Return rsync command to download all cubes or rss files for a plate
    Utils.prototype.rsyncFiles = function() {
    	var _this = this;
		$('.dropdown-menu').on('click','li a', function(){
						
			var id = $(this).attr('id');
			var plate = _this.GetURLParameter('plateid');
			var version = _this.GetURLParameter('version');
			var table = ($('#platetable .sastable').length == 0) ? null : $('#platetable .sastable').bootstrapTable('getData');

			// JSON request
			$.post($SCRIPT_ROOT + '/marvin/downloadFiles', {'id':id, 'plate':plate, 'version':version, 'table':JSON.stringify(table)},'json')
			.done(function(data){
				if (data.result.status == -1) {
					$('#rsyncbox').val('Error Message: '+data.result.message);
				} else {
					$('#rsyncbox').val(data.result.command);				
				}
			})
			.fail(function(data){
				$('#rsyncbox').val('Request for rsync link failed. Error: '+data.result.message);
			});
		});
    };

    // Submit username and login to Inspection DB for Trac login
    Utils.prototype.login = function(fxn) {
	  var form = $('#login_form').serialize();	
	  var fxn = this.getFunction();
      console.log('login fxn', fxn);
      var _this = this;
	  
	  $.post($SCRIPT_ROOT + '/marvin/login', form,'json') 
		  .done(function(data){
              console.log('logging in',data.result.status, data.result.ready);            
			  if (data.result.status < 0) {
				  // bad submit
				  _this.resetLogin();
			  } else {
				  // good submit
				  if (data.result.message != ''){
					  var stat = (data.result.status == 0) ? 'danger' : 'success'; 
					  htmlstr = "<div class='alert alert-"+stat+"' role='alert'><h4>" + data.result.message + "</h4></div>";
					  $('#loginmessage').html(htmlstr);
				  }
				  if (data.result['status']==1){
				  	  $('#inspectready').val(data.result.ready);
				  	  fxn.call(_this.object); 
				  }
				
			  }
		  })
		  .fail(function(data){
            alert('Bad login attempt');
		  });	
    };
    
    // Reset Login
    Utils.prototype.resetLogin	= function() {
		$('#loginform').modal('hide');
		$('#login_form').trigger('reset');	
		$('#loginmessage').empty();
    };

    // Retrieve the login function
    Utils.prototype.getFunction = function() {
        var fxnname = $('#fxn').val();
        if (fxnname == this.fxn.name) {
            return this.fxn;
        } else {
            console.error('Login function '+this.fxn.name+' does not match requested function '+fxnname);
            Raven.captureException('Login function '+this.fxn.name+' does not match requested function '+fxnname)
            return undefined;
        }
    }

    // Set the Utils login function
    Utils.prototype.setFunction = function(fxn, object) {
        this.fxn = fxn;
        this.object = object;
    }

    // Submit Login on Keyups
    Utils.prototype.submitLogin = function(event) {
        var _this = event.data;
		var fxn = _this.getFunction();

        // test for valid function
        if (_this.fxn === undefined) {
            msg = 'login fxn not defined';
            console.error(msg);
            Raven.captureException(msg);
        }

        // login
		if(event.keyCode == 13){
			if ($('#username').val() && $('#password').val()) {
				_this.login(_this.fxn);
			}
		}    	
    };

    // Focus Login
    Utils.prototype.focusLogin = function(event) {
    	$('#username').focus();
    };

    // Enable tooltips
    Utils.prototype.initToolTips = function() {
        $('[data-toggle="tooltip"]').tooltip();
    };

    // Enable bootstrap datetime-picker
    Utils.prototype.initDateTimePicker = function() {
        $('#datetimepicker').datetimepicker({
            widgetPositioning: {horizontal:'right'},
            format: 'YYYY-MM-DD'
        });
    };

    // Enable bootstrap select-picker
    Utils.prototype.enableSelectPicker = function() {
		$('.selectpicker').selectpicker();
    };

    // Get selected items from bootstrap select-picker
    Utils.prototype.getSelected = function(name) {
		var selectlist = [];
		var jname = name+' :checked';
		$(jname).each(function(){
			selectlist.push(this.value);
		});
		selectlist = (selectlist.length == 0) ? 'any' : selectlist;
		return selectlist;
    };

    // Initialize bootstrap tags
    Utils.prototype.initTags = function(name) {
		var tagbox = $(name).tags({
			tagData:[],
			tagSize:'sm',
			suggestions:[],
			promptText:'Enter a word or phrase and press Return',
			caseInsensitive: true
		});	
        return tagbox;        
    };

    // Reset bootstrap tags
    Utils.prototype.resetTags = function(name) {
		var tagbox = $(name).tags();
		var tags = tagbox.getTags();
		jQuery.each(tags, function(i,tag) {
			tagbox.removeTag(tag);
		});
		tagbox.removeLastTag();
    };

    // Name change for button collapse
    Utils.prototype.changeName = function(event) {
		if ($(this).hasClass('collapsed')) {
			$(this).button('reset');
		} else {
			$(this).button('complete');
		}
    };

    // Increment vote
    Utils.prototype.incrementVote = function(votediv) {
	    var count = parseInt($("~ .count", votediv).text());
	    console.log('inside incrementvote, votediv id', $(votediv).attr('id'));
	    if($(votediv).hasClass("up")) {
	      var count = count + 1;
	      $("~ .count", votediv).text(count);
	    } else {
	      var count = count - 1;
	      $("~ .count", votediv).text(count);     
	    }    
    };

    // Return unique elements of an Array
    Utils.prototype.unique = function(data) {
        var result = [];
        $.each(data, function(i, value) {
            if ($.inArray(value, result) == -1) result.push(value);
        });
        return result;
    };

    // Get the typeahead values
    Utils.prototype.getTypeahead = function() {
        //console.log('getting typeahead',query,$('#idtext').val());
        return ["Blah","hello","test"];
    };

	return Utils;
})();
