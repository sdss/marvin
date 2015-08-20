// Javascript code for general things

// Grab a GET parameter from the URL
function GetURLParameter(sParam){
	var sPageURL = window.location.search.substring(1);
	var sURLVariables = sPageURL.split('&');
	
	for (var i = 0; i < sURLVariables.length; i++){
		 var sParameterName = sURLVariables[i].split('=');
		 if (sParameterName[0] == sParam){
		 	return sParameterName[1];
		 }
	}
}

// Download all cubes or rss files for a plate (Returns rsync command)
function rsyncFiles(){
	$('.dropdown-menu').on('click','li a', function(){
					
		var id = $(this).attr('id');
		var plate = GetURLParameter('plateid');
		var version = GetURLParameter('version');
		var table = ($('#platetable .sastable').length == 0) ? null : $('#platetable .sastable').bootstrapTable('getData');

		// JSON request
		$.post($SCRIPT_ROOT + '/marvin/downloadFiles', {'id':id, 'plate':plate, 'version':version, 'table':JSON.stringify(table)},'json')
		.done(function(data){
			if (data.result.message !== null) {
				$('#rsyncbox').val('Error Message: '+data.result.message);
			} else {
				$('#rsyncbox').val(data.result.command);				
			}
		})
		.fail(function(data){
			$('#rsyncbox').val('Request for rsync link failed. Error: '+data.result.message);
		});
	});
						
}

// Submit username and password to Inspection DB for trac login
function login(fxn) {

  var form = $('#login_form').serialize();	
  var fxnname = $('#fxn').val();
  var fxn = window[fxnname];  
  
  $.post($SCRIPT_ROOT + '/marvin/login', form,'json') 
	  .done(function(data){
		  if (data.result['status'] < 0) {
			  // bad submit
			  resetLogin();
		  } else {
			  // good submit
			  if (data.result['message']!=''){
				  var stat = (data.result['status'] == 0) ? 'danger' : 'success'; 
				  htmlstr = "<div class='alert alert-"+stat+"' role='alert'><h4>" + data.result['message'] + "</h4></div>";
				  $('#loginmessage').html(htmlstr);
			  }
			  if (data.result['status']==1){
			  	  $('#inspectready').val(data.result.ready);
			  	  fxn.call(); 
			  }
			
		  }
	  })
	  .fail(function(data){
	  });				

}
	
// Reset Login
function resetLogin() {
	$('#loginform').modal('hide');
	$('#login_form').trigger('reset');	
	$('#loginmessage').empty();
}

// Set Focus to Username on loginform
$('#loginform').on('shown.bs.modal',function() {
	$('#username').focus();
});

// Submit loginform on username enter keypress
$(function() {		
	$('#username').keyup(function(event){
		var fxn = window[$('#fxn').val()];
		if(event.keyCode == 13){
			if ($('#username').val() && $('#password').val()) {
				login(fxn);
			}
		}
	});
});

// Submit loginform on password enter keypress
$(function() {
	$('#password').keyup(function(event){
		var fxn = window[$('#fxn').val()];
		if(event.keyCode == 13){
			if ($('#username').val() && $('#password').val()) {
				login(fxn);
			}
		}
	});
});

// Enable select picker
function enableSelectPicker() {
	$('.selectpicker').selectpicker();
}

// Get selected from picker
function getSelected(name) {
	var selectlist = [];
	var jname = name+' :checked';
	$(jname).each(function(){
		selectlist.push(this.value);
	});
	selectlist = (selectlist.length == 0) ? 'any' : selectlist;
	return selectlist;
}

// Initialize tags
function initTags(name) {
	var tagbox = $(name).tags({
		tagData:[],
		tagSize:'sm',
		suggestions:[],
		promptText:'Enter a word or phrase and press Return',
		caseInsensitive: true
	});	
}

// Reset tags
function resetTags(name) {
	var tagbox = $(name).tags();
	var tags = tagbox.getTags();
	jQuery.each(tags, function(i,tag) {
		tagbox.removeTag(tag);
	});
	tagbox.removeLastTag();
}

// Button Collapse Name Change Toggle
$(function() {
	$('.butcollapse').on('click',function() {
		if ($(this).hasClass('collapsed')) {
			$(this).button('reset');
		} else {
			$(this).button('complete');
		}
	});
});

function incrementVote(votediv) {
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

