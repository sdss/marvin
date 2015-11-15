
var TableMethods,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

TableMethods = (function() {

    marvin.TableMethods = TableMethods;

	// Constructor
    function TableMethods() {

        // in case constructor called without new
        if (false === (this instanceof TableMethods)) {
            return new TableMethods();
        }
        
        this.init();
        
    }

    // initialize the object
    TableMethods.prototype.init = function() {

    };

    // Write FITS file from table of Search Results [NOTE: MAY NOT BE USED]
    TableMethods.prototype.writeFits = function() {
		var data = $('.sastable').bootstrapTable('getData');
		var name = $('#fitsname').val();
		
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
    };

    // Delete Rows from table
    TableMethods.prototype.deleteRows = function() {
		var $table = $('.sastable');
		var $delete = $('#delete');	
			
		var ids = $.map($table.bootstrapTable('getSelections'), function (row) {
	        return row.id
	    });
	        
	    $table.bootstrapTable('remove', {
	        field: 'id',
	        values: ids
	    });
    };

    // Query parameters for paginaton
    TableMethods.prototype.queryParams = function() {
		return {
			type:'owner', sort:'updated', direction:'asc', per_page:100, page:1, pageSize:25
		};
    };

    // Determine color schemes for table cells
    TableMethods.prototype.cellStyle = function(value,row,index) {
		if (value == 'fault' || value == 'Out' || value == 'No' ){
			return { classes: 'danger'};
		} else if (value == 'In' || value == 'Yes'){
			return { classes: 'success'};
		}
		return {};	
    };

    // Sort on string versions of columns, needed in case of NULL values
    TableMethods.prototype.sort = function(a,b) {
		if (String(a) < String(b)) return 1;
		if (String(a) > String(b)) return -1;
		return 0;
    };

    return TableMethods;

})();

