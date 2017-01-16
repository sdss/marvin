/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2016-12-14 15:54:41
*/

// Javascript code for general things

'use strict';

class Utils {

    // Constructor
    constructor() {

    this.window = $(window);

    // login handlers
    $('#login-user').on('keyup', this, this.submitLogin); // submit login on keypress
    $('#login-pass').on('keyup', this, this.submitLogin); // submit login on keypress
    $('#login-drop').on('hide.bs.dropdown', this, this.resetLogin); //reset login on dropdown hide

    }

    // Print
    print() {
        console.log('I am Utils!');
    }

    // Build a Form
    buildForm(keys) {
        var args = Array.prototype.slice.call(arguments, 1);
        var form = {};
        keys.forEach(function (key, index) {
            form[key] = args[index];
        });
        return form;
    }

    // Serialize a Form
    serializeForm(id) {
        var form = $(id).serializeArray();
        return form;
    }

    // Unique values
    unique(data) {
        return new Set(data);
    }

    // Scroll to div
    scrollTo(location) {
        if (location !== undefined) {
            var scrolldiv = $(location);
            $('html,body').animate({scrollTop:scrolldiv.offset().top},1500, 'easeInOutExpo');
        } else {
            $('html,body').animate({scrollTop:0},1500, 'easeInOutExpo');
        }

    }

    // Initialize Info Pop-Overs
    initInfoPopOvers() {
        $('.infopop [data-toggle="popover"]').popover();
    };

    // Initialize tooltips
    initToolTips() {
        $('[data-toggle="tooltip"]').tooltip();
    };

    // Login function
    login() {
        var form = $('#loginform').serialize();
        var _this = this;

      $.post(Flask.url_for('index_page.login'), form, 'json')
          .done(function(data){
              if (data.result.status < 0) {
                  // bad submit
                  _this.resetLogin();
              } else {
                  // good submit
                  if (data.result.message !== ''){
                      var stat = (data.result.status === 0) ? 'danger' : 'success';
                      var htmlstr = "<div class='alert alert-"+stat+"' role='alert'><h4>" + data.result.message + "</h4></div>";
                      $('#loginmessage').html(htmlstr);
                  }
                  if (data.result.status === 1){
                      location.reload(true);
                  }

              }
          })
          .fail(function(data){
            alert('Bad login attempt');
          });
    };

    // Reset Login
    resetLogin() {
        $('#loginform').trigger('reset');
        $('#loginmessage').empty();
    };

    // Submit Login on Keyups
    submitLogin(event) {
        var _this = event.data;
        // login
        if(event.keyCode == 13){
            if ($('#login-user').val() && $('#login-pass').val()) {
                _this.login();
            }
        }
    };

    // Shows a banner
    marvinBanner(text, expiryDays, cookieName, url, urlText) {

        var _this = this;
        var expiryDays = (expiryDays === undefined) ? 0 : expiryDays;
        var cookieName = (cookieName === undefined) ? "marvin_banner_cookie" : cookieName;
        var url = (url === undefined) ? "" : url;
        var urlText = (urlText === undefined) ? "Learn more" : urlText;

        if (urlText == "" || url == "") {
            urlText = "";
            url = "";
        }

        _this.window[0].cookieconsent.initialise({
          "palette": {
            "popup": {
              "background": "#000"
            },
            "button": {
              "background": "#f1d600"
            }
          },
          "position": "top",
          "cookie": {
              "name": cookieName,
              "expiryDays": expiryDays,
              "domain": "localhost"},
          "content": {
              "message": text,
              "dismiss": 'Got it!',
              "href": url,
              "link": urlText}
        });

        if (expiryDays == 0) {
            document.cookie = cookieName + '=;expires=Thu, 01 Jan 1970 00:00:01 GMT;path=/;domain=localhost';
        };

    };

}
