
<script type='text/javascript'>
$(document).ready(function() {

	/*
		Generate sortable HTML table using D3 for 2d and 3d platelist 
	*/

	// Set the correct data for 2d or 3d stage
 	var stage = {{stage}};
 	stage = stage[0];
 	/*if (stage == '2d'){
 		var cols = {{keys2d}};
 		var data = {{plate2d}};
 		var flags = {{flags2d}};
 	} else {
 		var cols = {{ keys3d }};
 		var data = {{plate3d}};
 		var flags = {{flags3d}};
 	}*/
 	var cols = {{keys}};
 	var data = {{table}};
 	var flags = {{flags}};

	// Do table call
	var table = tabulate(data, cols);

	// HTML table version
	
	// Sorting functions		
	function sortRows(type,column){

    	var sort;
    	var sortValueAscending = function(a, b) { return d3.ascending(a[column], b[column]) };
    	var sortValueDescending = function(a, b) { return d3.descending(a[column], b[column]) };    

    	if (type) sort = sortValueAscending;
    	else sort = sortValueDescending;
    	return sort;
	}

	// Build the table
	function tabulate(data, columns) {
    	var table = d3.select("#viz").append("table").attr('class','table table-bordered sastable').style('border-collapse','collapse')
    		.style('border','2 px solid').attr('id','pltable');
        var thead = table.append("thead");
        var tbody = table.append("tbody");
    
    	var metricAscending = true;

    	// Append the header row
    	thead.append("tr").attr('id','head').selectAll("th").data(columns).enter().append("th")
            .text(function(column) { return column.toUpperCase(); })
            .style('display', function(column) {return column=='image' ? 'None' : ''})
        	.on("click", function (column) {
            	var sort;
			
            	// Choose appropriate sorting function.
            	sort = sortRows(metricAscending, column);
            	metricAscending = !metricAscending;
            
            	var rows = tbody.selectAll("tr").sort(sort);
        	});

    	// Create a row for each object in the data
    	var rows = tbody.selectAll("tr").data(data).enter().append("tr"); 

		// make tooltip
		tooltip = d3.selectAll('#viz').append("div")
			.attr("class", "tooltip")
			.style(0);
						    
    	// Create a cell in each row for each column
    	var cells = rows.selectAll("td")
        	.data(function(row) {
            	return columns.map(function(column) {
                	return {column: column, value: row[column], plate:row['plate']};
            	});
        	})
        	.enter()
        	.append("td")
            .text(function(d) { return d.column=='apocomp' ? ' ' : d.value; })
            .attr('id',function(d) {return d.column;})
            .attr('class', function(d) {return d.value=='fault' || d.value=='Out' || (d.column == 'complete' && d.value=='No') ? 'danger' : d.value=='In' || d.value=='Yes' ? 'success' : '';})
            .style('display', function(d) {return d.column=='image' ? 'None' : '';})
            .html(function(d) {
            	if (stage=='2d'){
            		var htmlref = d.column=='plate' ? d.value : d.column=='mjd' ? d.plate+"/"+d.value : '';
            	} else {
            	    var imgref = d.column=='image' ? d.plate+'/stack/images/'+d.value : '';
            		var htmlref = d.column=='plate' ? d.value+'/stack' : '';
            		var imlink = "<img src='"+imgref+"' width='42' height='42'>";
            		var htmllink = "<a href='"+imgref+"' target='_blank'>"+imlink+"</a>";
            	}
            	return d.column=='plate' || d.column=='mjd' ? "<a href='"+htmlref+"' target='_blank'>"+d.value+"</a>" : (d.column=='image' && d.value.trim() != 'NULL') ? htmllink : d.value;
            	})
            .on('mouseover', function(d,i,j){
            	// build the tooltip table
            	var isstring = typeof(flags[j])==='string';
            	if (isstring==true){
            		tipstring = '<table class="table table-bordered table-condensed"><tr><th>Quality Flags</th></tr><tr><td>'+flags[j]+'</td></tr></table>';
            	} else {
            		tipstring = '<table class="table table-bordered table-condensed"><tr><th>Quality Flags</th></tr>';
            		for (f=0;f<flags[j].length;f++){
            			tipstring+='<tr><td>'+flags[j][f]+'</td></tr>';
            		}
            		tipstring += '</table>';
            	}
            	
            	// actual transition and loading of tooltip
            	if (d.column=='drp2qual' || d.column=='drp3qual'){
            	tooltip.transition().duration(200).style('opacity',1.0);            	
            	tooltip.html(tipstring)
            		.style("left", (d3.event.pageX+10) + "px")
            		.style("top", (d3.event.pageY+10) + "px")
            		.style('background-color', 'lightgray');
            	}	
            })
            .on("mouseout", function(d) {       
            	tooltip.transition()        
                .duration(500)      
                .style("opacity", 0);
                tooltip.selectAll('table').remove();
            });   	

		// Toggle button
		var toggle = d3.select('#imgtoggle');
		var togval = false;
		toggle.on('click', function(d,i){ 
			
			togval = !togval;
			if (togval==true){
				thead.selectAll('tr').selectAll('th').data(columns).style('display', function(column){return column=='image' ? '' : '';});
				rows.selectAll('td').data(function(row) {
            								return columns.map(function(column) {
                								return {column: column, value: row[column], plate:row['plate']};
            								});})
            						.style('display', function(d){return d.column=='image' ? '' : '';});
				
			} else {
				thead.selectAll('tr').selectAll('th').data(columns).style('display', function(column){return column=='image' ? 'None' : '';});
				rows.selectAll('td').data(function(row) {
            								return columns.map(function(column) {
                								return {column: column, value: row[column], plate:row['plate']};
            								});})
            						.style('display', function(d){return d.column=='image' ? 'None' : '';});
			}
			
			return console.log('toggling images'); 
			});
    	
    	return table;
	}
		

});
</script>