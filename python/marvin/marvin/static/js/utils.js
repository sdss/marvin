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
		var plate = GetURLParameter('plateID');
		var version = GetURLParameter('version');
		var table = ($('.sastable').length == 0) ? null : $('.sastable').bootstrapTable('getData');

		// JSON request
		$.post($SCRIPT_ROOT + '/marvin/downloadFiles', {'id':id, 'plate':plate, 'version':version, 'table':JSON.stringify(table)},'json')
		.done(function(data){
			$('#rsyncbox').val(data.result);
		})
		.fail(function(data){
			$('#rsyncbox').val('Request for rsync link failed.');
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
			  	  console.log('after login,ready',data.result.ready);
			  	  console.log('before',$('#inspectready').val());
			  	  $('#inspectready').val(data.result.ready);
			  	  console.log('after',$('#inspectready').val());
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
