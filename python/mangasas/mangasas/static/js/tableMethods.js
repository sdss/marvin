
// Write FITS
function writeFits(){
	var data = $('.sastable').bootstrapTable('getData');
	var name = $('#fitsname').val();
	/*$.getJSON($SCRIPT_ROOT + '/writeFits', {'data':JSON.stringify(data),'name':name}, 
		function(data){
			$('#fitsout').text(data.result);
		});*/
		
	/*$.get($SCRIPT_ROOT + '/writeFits', {'data':JSON.stringify(data),'name':name},'xml')
		.done(function(data){
			console.log(data);
			$('#fitsout').text('data success');
		})
		.fail(function(data){
			$('#fitsout').text('data fail');
		});*/
	
	console.log('name',name);
	console.log('data',data);	
	var request = $.ajax({
		url: $SCRIPT_ROOT + '/writeFits',
		data: {'data':JSON.stringify(data),'name':name},
		type: 'POST'
	});
	request.done(function(data){
		//console.log(data);
		//alert(request.getAllResponseHeaders());
		//$('#fitsout').text('data success');
		alert('success');
		$('#download').submit();
	});
	request.fail(function(data){
		alert('failed');
	});
	$('#download').submit(function(event){
		alert('form submitted');
	});
}
	
// Delete rows from data table
function deleteRows(){
	var $table = $('.sastable');
	var $delete = $('#delete');	
		
	var ids = $.map($table.bootstrapTable('getSelections'), function (row) {
        return row.id
    });
        
    $table.bootstrapTable('remove', {
        field: 'id',
        values: ids
    });
}
	
// Query params for pagination
function queryParams(){
	return {
		type:'owner', sort:'updated', direction:'asc', per_page:100, page:1, pageSize:25
	};
}

// Determine color scheme for cells (faults, etc)	
function cellStyle(value,row,index){
		
	if (value == 'fault' || value == 'Out' || value == 'No' ){
		return { classes: 'danger'};
	} else if (value == 'In' || value == 'Yes'){
		return { classes: 'success'};
	}
	return {};	
}

// Sort on string versions, needed in case of NULL values	
function sort(a,b){
	if (String(a) < String(b)) return 1;
	if (String(a) > String(b)) return -1;
	return 0;
}
	
