/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 00:10:26
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-05-23 17:36:14
*/

// Javascript code for general things
//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Utils = function () {

    // Constructor
    function Utils() {
        _classCallCheck(this, Utils);

        this.window = $(window);

        // login handlers
        $('#login-user').on('keyup', this, this.submitLogin); // submit login on keypress
        $('#login-pass').on('keyup', this, this.submitLogin); // submit login on keypress
        $('#login-drop').on('hide.bs.dropdown', this, this.resetLogin); //reset login on dropdown hide
    }

    // Print


    _createClass(Utils, [{
        key: 'print',
        value: function print() {
            console.log('I am Utils!');
        }

        // Build a Form

    }, {
        key: 'buildForm',
        value: function buildForm(keys) {
            var args = Array.prototype.slice.call(arguments, 1);
            var form = {};
            keys.forEach(function (key, index) {
                form[key] = args[index];
            });
            return form;
        }

        // Serialize a Form

    }, {
        key: 'serializeForm',
        value: function serializeForm(id) {
            var form = $(id).serializeArray();
            return form;
        }

        // Unique values

    }, {
        key: 'unique',
        value: function unique(data) {
            return new Set(data);
        }

        // Scroll to div

    }, {
        key: 'scrollTo',
        value: function scrollTo(location) {
            if (location !== undefined) {
                var scrolldiv = $(location);
                $('html,body').animate({ scrollTop: scrolldiv.offset().top }, 1500, 'easeInOutExpo');
            } else {
                $('html,body').animate({ scrollTop: 0 }, 1500, 'easeInOutExpo');
            }
        }

        // Initialize Info Pop-Overs

    }, {
        key: 'initInfoPopOvers',
        value: function initInfoPopOvers() {
            $('.infopop [data-toggle="popover"]').popover();
        }

        // Initialize tooltips

    }, {
        key: 'initToolTips',
        value: function initToolTips() {
            $('[data-toggle="tooltip"]').tooltip();
        }

        // Select Choices from a Bootstrap-Select element

    }, {
        key: 'selectChoices',
        value: function selectChoices(id, choices) {
            $(id).selectpicker('val', choices);
            $(id).selectpicker('refresh');
        }

        // Reset Choices from a Bootstrap-Select element

    }, {
        key: 'resetChoices',
        value: function resetChoices(id) {
            console.log('reseting in utils', id);
            var select = typeof id === 'string' ? $(id) : id;
            select.selectpicker('deselectAll');
            select.selectpicker('refresh');
            select.selectpicker('render');
        }

        // Login function

    }, {
        key: 'login',
        value: function login() {
            var _this2 = this;

            var form = $('#loginform').serialize();
            Promise.resolve($.post(Flask.url_for('index_page.login'), form, 'json')).then(function (data) {
                if (data.result.status < 0) {
                    throw new Error('Bad status login');
                }
                if (data.result.message !== '') {
                    var stat = data.result.status === 0 ? 'danger' : 'success';
                    var htmlstr = '<div class=\'alert alert-' + stat + '\' role=\'alert\'><h4>' + data.result.message + '</h4></div>';
                    $('#loginmessage').html(htmlstr);
                }
                if (data.result.status === 1) {
                    location.reload(true);
                }
            }).catch(function (error) {
                _this2.resetLogin();
                alert('Bad login attempt');
            });
        }

        // Reset Login

    }, {
        key: 'resetLogin',
        value: function resetLogin() {
            console.log('reset');
            $('#loginform').trigger('reset');
            $('#loginmessage').empty();
        }

        // Submit Login on Keyups

    }, {
        key: 'submitLogin',
        value: function submitLogin(event) {
            var _this = event.data;
            // login
            if (event.keyCode == 13) {
                if ($('#login-user').val() && $('#login-pass').val()) {
                    _this.login();
                }
            }
        }

        // Shows a banner

    }, {
        key: 'marvinBanner',
        value: function marvinBanner(text, expiryDays, cookieName, url, urlText) {

            var _this = this;
            expiryDays = expiryDays === undefined ? 0 : expiryDays;
            cookieName = cookieName === undefined ? "marvin_banner_cookie" : cookieName;
            url = url === undefined ? "" : url;
            urlText = urlText === undefined ? "Learn more" : urlText;

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
                    "domain": "localhost" },
                "content": {
                    "message": text,
                    "dismiss": 'Got it!',
                    "href": url,
                    "link": urlText }
            });

            if (expiryDays === 0) {
                document.cookie = cookieName + '=;expires=Thu, 01 Jan 1970 00:00:01 GMT;path=/;domain=localhost';
            }
        }
    }]);

    return Utils;
}();
