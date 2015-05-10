
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
		$.post($SCRIPT_ROOT + '/manga/downloadFiles', {'id':id, 'plate':plate, 'version':version, 'table':JSON.stringify(table)},'json')
		.done(function(data){
			$('#rsyncbox').val(data.result);
		})
		.fail(function(data){
			$('#rsyncbox').val('Request for rsync link failed.');
		});
	});
						
}

