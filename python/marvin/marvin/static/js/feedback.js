

var Feedback,
  __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

Feedback = (function () {

    function Feedback() {

        // in case constructor called without new
        if (false === (this instanceof Feedback)) {
            return new Feedback();
        }
        
        this.init();
        this.setLoginFxn();
        
        // Event Handlers
        
        // capture the row id
        $(document).on('click', '#table_feedback tr', this, this.getRowId);
        // send state changes on statuses and votes
        $(document).on('click', '.increment', this, this.changeState);
        $(document).on('change', '.feedback_status', this, this.changeState);

    }
    
    // initialize the object
    Feedback.prototype.init = function() {
        this.rowid = null;
        this.tableindex = null;
        this.dataindex = null;
        this.fxn = 'reloadPage';
    };

    // test print
    Feedback.prototype.print = function() {
        console.log('We are now printing feedback for ',this.rowid);
    };

    // Reload the page
    Feedback.prototype.reloadPage = function reloadPage() {
        location.reload(true);
    };

    // Set the Feedback login function
    Feedback.prototype.setLoginFxn = function() {
        $('#fxn').val(this.fxn);
        utils.setFunction(this.reloadPage, this);
    };

    // Get the Row ID 
    Feedback.prototype.getRowId = function(event) {
        var _this = event.data;
        _this.dataindex = this.rowIndex;
        var children = $(this).children('td');
        if (children.length > 0) {
            _this.rowid = parseInt($(this).children('td')[1].textContent);
            _this.tableindex = _this.rowid - 1;
        }
    };

    // Send vote and status changes to server with Ajax
    Feedback.prototype.changeState = function(event) {
        var _this = event.data;
        var status = $(this).find(':selected').text();
        var vote = (status) ? null : $(this).hasClass('up') ? 1 : -1;
        var count = (status) ? null : $("~ .count", this).text();
        var type = (status) ? 'status' : 'vote';

        var form = {'id':_this.rowid, 'status':status, 'vote':vote, 'type':type};

        _this.sendAjax(form, $(this));
    };

    // Send feedback Ajax request
    Feedback.prototype.sendAjax = function(form, div) {
        var url = '/marvin/feedback/'+form.type+'/update';
        var _this = this;
        console.log('sending ajax',$SCRIPT_ROOT + url);
        $.post($SCRIPT_ROOT + url, form, 'json')
            .done(function(data) {
                if (data.result['status'] == 1) {
                    // reload if vote change
                    if (form.type === 'vote') {
                        utils.incrementVote(div);
                        _this.reloadPage();
                    }
                } else {
                    alert('Inspection failed with message: '+data.result['message']);
                }
            })
            .fail(function(data) {
                alert('Error in repsonse from Inspection webapp. Please contact admin@sdss.org')
            });
    };

    // Promote Trac Ticket
    Feedback.prototype.promoteTracTicket = function(id) {
        var _this = this;
        $('#promotemessage').html('Promoting your ticket. Please wait...');
        console.log('promoting trac',$SCRIPT_ROOT + '/marvin/feedback/tracticket/promote');
        $.post($SCRIPT_ROOT + '/marvin/feedback/tracticket/promote', {'id':id}, 'json')
            .done(function(data){
                if (data.result['status'] == 1) {
                  _this.reloadPage();
                } else {
                  alert('Error promoting ticket (returned status != 1): '+data.result['message']);
                }
            })
            .fail(function(data){
                alert("Error in response from Inspection webapp.  Please contact admin@sdss.org");
            })  
            .always(function(data){
                $('#promotemessage').html('');
            });     
    };

    return Feedback;

})();


