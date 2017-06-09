/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-05-23 17:36:14
*/

// Javascript code for general things
//jshint esversion: 6
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
        const args = Array.prototype.slice.call(arguments, 1);
        const form = {};
        keys.forEach((key, index)=>{ form[key] = args[index]; });
        return form;
    }

    // Serialize a Form
    serializeForm(id) {
        const form = $(id).serializeArray();
        return form;
    }

    // Unique values
    unique(data) {
        return new Set(data);
    }

    // Scroll to div
    scrollTo(location) {
        if (location !== undefined) {
            const scrolldiv = $(location);
            $('html,body').animate({scrollTop:scrolldiv.offset().top},1500, 'easeInOutExpo');
        } else {
            $('html,body').animate({scrollTop:0},1500, 'easeInOutExpo');
        }

    }

    // Initialize Info Pop-Overs
    initInfoPopOvers() {
        $('.infopop [data-toggle="popover"]').popover();
    }

    // Initialize tooltips
    initToolTips() {
        $('[data-toggle="tooltip"]').tooltip();
    }

    // Select Choices from a Bootstrap-Select element
    selectChoices(id, choices) {
      $(id).selectpicker('val', choices);
      $(id).selectpicker('refresh');
    }

    // Reset Choices from a Bootstrap-Select element
    resetChoices(id) {
      console.log('reseting in utils', id);
      let select = (typeof id === 'string') ? $(id) : id;
      select.selectpicker('deselectAll');
      select.selectpicker('refresh');
      select.selectpicker('render');
    }

    // Login function
    login() {
        const form = $('#loginform').serialize();
        Promise.resolve($.post(Flask.url_for('index_page.login'), form, 'json'))
          .then((data)=>{
              if (data.result.status < 0) {
                throw new Error('Bad status login');
              }
              if (data.result.message !== ''){
                  const stat = (data.result.status === 0) ? 'danger' : 'success';
                  const htmlstr = `<div class='alert alert-${stat}' role='alert'><h4>${data.result.message}</h4></div>`;
                  $('#loginmessage').html(htmlstr);
              }
              if (data.result.status === 1) {
                location.reload(true);
              }
            })
          .catch((error)=>{
            this.resetLogin();
            alert('Bad login attempt');
          });

      }

    // Reset Login
    resetLogin() {
        console.log('reset');
        $('#loginform').trigger('reset');
        $('#loginmessage').empty();
    }

    // Submit Login on Keyups
    submitLogin(event) {
        const _this = event.data;
        // login
        if(event.keyCode == 13){
            if ($('#login-user').val() && $('#login-pass').val()) {
                _this.login();
            }
        }
    }

    // Shows a banner
    marvinBanner(text, expiryDays, cookieName, url, urlText) {

        const _this = this;
        expiryDays = (expiryDays === undefined) ? 0 : expiryDays;
        cookieName = (cookieName === undefined) ? "marvin_banner_cookie" : cookieName;
        url = (url === undefined) ? "" : url;
        urlText = (urlText === undefined) ? "Learn more" : urlText;

        if (urlText === "" || url === "") {
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

        if (expiryDays === 0) {
            document.cookie = cookieName + '=;expires=Thu, 01 Jan 1970 00:00:01 GMT;path=/;domain=localhost';
        }

    }

}
