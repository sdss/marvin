// Javascript code for Marvin DAP QA plots/comments on individual plates

// toggle DAP categories
$(function() {
	$('#dapqacat_select').change(function() {
		var key = $(this).val();
		// show category options
		$('.dapqacatoptions').hide();
		$('#dapqacatoptions_'+key).show();
		// show option reminder
		$('.qacomment').html('');
		// set default selection
		setDefault(key);
	});
});


// reset dapqa form on page load
/*$(function() {
	$('.dapqacomment_form').trigger('reset');
	$('.qacomment').html('');
	// load a default
	var key = 'maps';
	setDefault(key);
		
	var dapqa = new Dapqa('7443','9101','10');
	if (typeof dapqa != 'undefined') {
	console.log('dapqa',dapqa);
	dapqa.print();
	console.log('mainpane',dapqa.mainpane);
	console.log('qaform',dapqa.qaform.find('#dapqacat'));
	}
});*/

// dapqa tab selection
$(function(){
	$('#cubetabs a').click(function() {	
		var ifu = getIFUHash().slice(1);
		var dapifuform = $('#dapqacomment_form_'+ifu);
		dapifuform.trigger('reset');
		$('.qacomment').html('');
		var key = 'maps';
		//var test = "{{platera|tojson|safe}}";
		//console.log('ready',test);
		setDefault(key);		
	});
});
	
// store old values
function storeold(key,mapid,qatype) {
	$('#oldmapid').val(mapid);
	$('#oldqatype').val(qatype);
	$('#oldkey').val(key);
}

// set category default
function setDefault(key) {
	//var html = (key!='radgrad') ? 'cube-none2': 'rss-rad1';
	var html = (key == 'maps') ? 'cube-none2' : (key =='spectra') ? 'cube-all5' : 'rss-rad1';
	var mapid = (key=='maps') ? 'kin' : (key=='radgrad') ? 'emflux': 'spec0'; 
	$('#qacomment_'+key).html(html);
	
	// display list and panels, store old values
	displayList(key);
	if (key=='spectra') getSpectraList(key,mapid,html);
	getPanel(key,mapid,html);
	storeold(key,mapid,html)	
}

// toggle DAP QA cube/rss
$(function() {
	$('.dropdown-menu.qalist').on('click','li a', function() {
		var id = $(this).attr('id');
		var parentid = $(this).parent().parent().attr('id');
		var type = parentid.split('typelist')[0];
		var key = parentid.split('_')[1];

		// build html reference
		var html = $('#qacomment_'+key).html();
		html = type+'-'+id;
		$('#qacomment_'+key).html(html);
		
		//display the appropriate list
		displayList(key);

		// get map id
		var mapid = $('#catlist_'+key+' select option:selected').attr('id');

		//for spectra, populate list for first time
		if (key=='spectra') getSpectraList(key,mapid,html);
		
		//get new panel and store old values
		getPanel(key,mapid,html);
		storeold(key,mapid,html);
			
	});
});

// Display the appropriate list and reset selection
function displayList(key) {
	$('.catlist').hide();
	$('#catlist_'+key).show();
	$('#catlist_'+key+' select').children().removeProp('selected');
	
	//set first option as default
	var first = $('#catlist_'+key+' select :first-child');
	first.prop('selected',true);
}

// toggle DAP map selection
$(function() {
	$('.daptoggle').change(function() {
		var id = $(this).attr('id');
		var key = (id.search('map') != -1) ? 'maps' : (id.search('spectra') != -1) ? 'spectra' : (id.search('radgrad') != -1) ? 'radgrad' : ''
		var mapid = $('#'+id+' option:selected').attr('id');
		var qatype = $('#qacomment_'+key).html();

		// get old values
		var oldmapid = $('#oldmapid').val();
		var oldqatype = $('#oldqatype').val();
		var oldkey = $('#oldkey').val();
		console.log('map select id,key,mapid',id,key,mapid,oldmapid,oldqatype,oldkey);

		// get new panel and store old values
		getPanel(key,mapid,qatype);
		storeold(key,mapid,qatype);
	});
});

// build the DAP form
function buildDAPform(newdata=null,ifu=null) {
	var dapform = $('#dapqacomment_form_'+ifu).serializeArray();
	if (newdata) {
		$.each(newdata,function(i,val) {
			dapform.push(val);
		});
	}
	return dapform
}

// parse DAP issues
function parseDAPissues(key=null) {
	var name = (key) ? 'select[id*="dapqa_issue_'+key+'"]' : '.dapqaissuesp';
	var issuelist = getSelected(name);
	return issuelist;
}

// get a DAP panel
function getPanel(key, mapid, qatype) {
	// key = category key
	// qatype = category options from dropdowns
	// mapid = id of option from list selection

	$('.dapqapanel').hide();
	var ifu = $('#ifu').val();
	console.log('ifu field',$('#ifu'));
	var ifuname = $('#ifuname').val();
	
	// build form data
	issues = parseDAPissues(key);
	newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'cubepk','value':$('#cubepk').val()},
			   {'name':'qatype','value':qatype},{'name':'issues','value':JSON.stringify(issues)}];
	dapformdata = buildDAPform(newdata=newdata,ifu=ifu)
	console.log('dapform',dapformdata);	

	
	$.post($SCRIPT_ROOT + '/marvin/getdappanel', dapformdata,null,'json')
		.done(function(data){
			console.log('ifu',ifu,ifuname);
			var ifupanel = $('#dapqapane_'+ifu).find('#dapqa_'+key);
			console.log(ifupanel);
			ifupanel.show();
			var title = $('#dapqa_'+key+' h4');
			if (data.result['title']) title.html(data.result['title']);
			
			// setsession status failure
			if (data.result['setsession'] && data.result['setsession']['status'] == 0) {
				var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+data.result['setsession']['message']+"</h4></div>";
				title.html(alerthtml);
			}
			
			loadImages(key,data.result['images'],data.result['panelmsg']);
			loadComments(key,data.result['getsession']);
			
		})
		.fail(function(data){
			$('#dapqa_'+key).show();
			var title = $('#dapqa_'+key+' h4');
			var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to retrieve data!</h4></div>";
			title.html(alerthtml);
		});	
	
}

// get list of DAP spectrum plots available
function getSpectraList(key,mapid,qatype) {

	newdata = [{'name':'key','value':key},{'name':'mapid','value':mapid},{'name':'qatype','value':qatype}];
	dapformdata = buildDAPform(newdata=newdata)
			
	$.post($SCRIPT_ROOT + '/marvin/getdapspeclist', dapformdata,null,'json')
		.done(function(data){
			
			var speclist = data.result['speclist'];
			$('#dapspectralist').empty();
			if (speclist) {
				$.each(speclist,function(i,name) {
					
					var specname = name.split('-')[0];
					var specnum = name.split('-')[1];
					specnum = specnum.replace(/^0+/, '');
					if (specnum.length == 0) specnum='0';
					var id = specname+specnum
					
					$('#dapspectralist').append($('<option>', { 
						id: id,
						value: id,
						text : name,
						selected : (i==0) ? true : false 
					}));
				});
			}
		})
		.fail(function(data){
			$('#dapqa_'+key).show();
			var title = $('#dapqa_'+key+' h4');
			var alerthtml = "<div class='alert alert-danger' role='alert'><h4>Server Error: Failed to retrieve list of spectra!</h4></div>";
			title.html(alerthtml);
		});	
} 

// load DAP plot images
function loadImages(key,images, msg) {
	$('#dapqa_'+key+' img').removeProp('src');
	//console.log($('#dapqa_'+key+' img'));
	if (images) {
		$('#dapqa_'+key+' img').each(function(index) {
			//console.log($(this),index,images.length);
			$(this).attr('src',images[index]);
			//$(this).magnify();
		});	
	} else {
		var title = $('#dapqa_'+key+' h4');
		var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+msg+"</h4></div>";
		title.html(alerthtml);		
	}
}

// reset and load all the issues and comments
function loadComments(key,results) {
	// reset all issues
	
	console.log('loadComments results', results);
	
	if (results['status'] == 1) {
    
        // load new comments
        console.log("Loading results for key="+key+"..."); //delete this print (just for debug)
        $.each(results['dapqacomments'],function(i,panelcomment) {
        
            //replace this print with actual javascript to replace the panel comments with panelcomment in this loop over results
            //print "Brian, please loadcomment for panelname="+panelcomment['panel']+", panelnumber="+panelcomment['position']+", comment="+panelcomment['comment']+", issueids="+panelcomment['issues']+"!"
            
            console.log('panelcomment',panelcomment);
            $('#dapqa_comment'+panelcomment.catid+'_'+panelcomment.position).val(panelcomment.comment);
            
            $.each(panelcomment.issues, function(i,issid) {
            	//dapqa_issue_{{category.key}}{{"_"+mapnum if mapnum else ""}}
            	//issue_{{issid}}{{"_"+mapnum if mapnum else ""}}
            	$('#issue_'+issid+'_'+panelcomment.position).prop('selected',true);
            }
        });
	
	} else {
		// getsession status failure
		var title = $('#dapqa_'+key+' h4');
		var alerthtml = "<div class='alert alert-danger' role='alert'><h4>"+results['message']+"</h4></div>";
		title.html(alerthtml);	
	}
	
}


// Map magnify toggle	
$(function(){
	$('#magtoggle').on('click',function(){
		if ($(this).hasClass('active')){
			//$('.dapqapanel img').parent().removeClass('.magnify');
		} else {
			$('.dapqapanel img').magnify();
		}
	});
});

// Load the DAP image modal
function daploadmodal(img) {
	var src = img.src;
	var name = src.slice(src.search('manga-'));
	$('#dapimgtitle').html(name);
	var image = '<img class="img-responsive img-rounded" src="'+src+'" alt="Image"/>';
	$('#dapimgbody').html(image);
}

// Submit DAP QA Comments
function dapaddcomments(ifu) {
	var dapform = $('#dapqacomment_form_'+ifu);
	console.log(dapform.serialize());
}




