
<script type='text/javascript'>

function loadPlateDesign(){

	var plateid = {{plateid}};
	var plateCen = {'ra': {{platera}}, 'dec': {{platedec}} };
	var width = 400;
	var height = 400;
	var padding = 55;

	var data = {{platedata}};
 
 	// make scales
 		// image scales
	var xscale = d3.scale.linear().domain([330.,-330.]).range([0,400]);
	var yscale = d3.scale.linear().domain([330.,-330.]).range([0,400]);
	
		// axis scales - static whole and dynamic
	var fullrascale = d3.scale.linear().domain([Math.ceil(plateCen.ra+1.5),Math.floor(plateCen.ra-1.5)]).range([0,400]);
	var fulldecscale = d3.scale.linear().domain([plateCen.dec+1.5,plateCen.dec-1.5]).range([0,400]);		
	var rascale = d3.scale.linear().domain([plateCen.ra+1.5,plateCen.ra-1.5]).range([0,400]);
	var decscale = d3.scale.linear().domain([plateCen.dec+1.5,plateCen.dec-1.5]).range([0,400]);
		
	// make the main svg element	
	var svg = d3.select('#platedesign').append('svg')
		.attr('width',width+padding)
		.attr('height',height+padding);

	// initial focus and view
	var focus = {'name':7443,'color': 'white', 'cx': 0.0, 'cy': 0.0, 'r': 200};
	var view = [focus.cx,focus.cy,focus.r*2];
	
	// make the tooltip
	tooltip = d3.select('#platedesign').append('div')
		.attr('class','tooltip')
		.style('opacity',0);
		
	// add the plate and ifu data	
	var ifus = svg.selectAll('circle').data(data).enter().append('circle')
		.attr('class', function(d){return d.name != plateid ? 'ifu' : '';})
		.attr('transform',function(d,i){return 'translate(' + xscale(d.cx) + "," + yscale(d.cy) + ')';})
		.attr('r',function(d,i){return d.r;})
		.style('fill',function(d,i){return d.color;})
		.style('stroke','black')
		.on('mouseover', function(d){
			if (d.name != plateid){
			tooltip.transition()
				.duration(200)
				.style('opacity',0.9);
			tooltip.html(d.name + '<br/> RA: '+d.ra+'<br/> Dec: ' +d.dec)
				.style('left',(d3.event.pageX-800)+'px')
				.style('top', (d3.event.pageY-200)+'px');
			}
		})
		.on('mouseout',function(d){
			tooltip.transition()
				.duration(500)
				.style('opacity',0);
		})
		.on('click',function(d){
			if (focus != d) zoom(d), d3.event.stopPropagation();
		});
		
		
	// add the axes	
	xaxis = d3.svg.axis().scale(rascale).orient('bottom').ticks(5);
	yaxis = d3.svg.axis().scale(decscale).orient('right').ticks(5);
	svg.append('g').attr('class','x axis')
		.attr('transform','translate(0,'+(height+5)+')')
		.call(xaxis)
		.append('text')
		.attr('x',width/2)
		.attr('y',35)
		.style('text-anchor','middle')
		.text('RA');
		
	svg.append('g').attr('class','y axis')
		.attr('transform','translate('+(width+5)+',0)')
		.call(yaxis)
		.append('text')
		.attr('transform','rotate(90)')
		.attr('x',height/2)
		.attr('y',-35)
		.style('text-anchor','middle')
		.text('Dec');
	// zooming controls
	var node = svg.selectAll('circle');
		
	function zoom(d){
		var focus0 = focus; focus=d;
		var newview = [d.cx, d.cy, d.r * 2 + 20];
		
		var transition = d3.transition()
			.duration(d3.event.altKey ? 7500 : 750)
			.tween('zoom', function(d){
				var i = d3.interpolateZoom(view, newview);
				return function(t) {zoomTo(i(t)); };
			});
		
	}

	function zoomTo(v) {
		// zoom circles
    	var k = height / v[2]; view = v;
    	node.attr("transform", function(d) { return "translate(" + xscale((d.cx - v[0])*k) + "," + yscale((d.cy - v[1])*k) + ")"; });
    	ifus.attr("r", function(d) { return d.r * k; });
    	
		// zoom axes
		var newra = [fullrascale.invert(xscale(v[0])-v[2]/2),fullrascale.invert(xscale(v[0])+v[2]/2)];
		var newdec = [fulldecscale.invert(yscale(v[1])-v[2]/2), fulldecscale.invert(yscale(v[1])+v[2]/2)];
		rascale.domain(newra);
		decscale.domain(newdec);    	
    	svg.select(".x.axis").call(xaxis);
  		svg.select(".y.axis").call(yaxis);    	
  	}	

		

}
</script>

