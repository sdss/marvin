/*
* @Author: Brian Cherinka
* @Date:   2016-04-29 09:29:24
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:11:43
*/

//jshint esversion: 6
'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Carousel = function () {

    // Constructor
    function Carousel(cardiv, thumbs) {
        _classCallCheck(this, Carousel);

        this.carouseldiv = $(cardiv);
        this.thumbsdiv = thumbs !== undefined ? $(thumbs) : $('[id^=carousel-selector-]');

        // init the carousel
        this.carouseldiv.carousel({
            interval: 5000
        });

        // Event handlers
        this.thumbsdiv.on('click', this, this.handleThumbs);
        this.carouseldiv.on('slid.bs.carousel', this, this.updateText);
    }

    // Print


    _createClass(Carousel, [{
        key: 'print',
        value: function print() {
            console.log('I am Carousel!');
        }

        // Handle the carousel thumbnails

    }, {
        key: 'handleThumbs',
        value: function handleThumbs(event) {
            var _this = event.data;
            var id_selector = $(this).attr("id");
            try {
                var id = /-(\d+)$/.exec(id_selector)[1];
                //console.log(id_selector, id);
                _this.carouseldiv.carousel(parseInt(id));
            } catch (e) {
                console.log('MyCarousel: Regex failed!', e);
            }
        }

        // When carousel slides, auto update the text

    }, {
        key: 'updateText',
        value: function updateText(event) {
            var _this = event.data;
            var id = $('.item.active').data('slide-number');
            $('#carousel-text').html($('#slide-content-' + id).html());
        }
    }]);

    return Carousel;
}();
