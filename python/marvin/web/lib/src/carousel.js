/*
* @Author: Brian Cherinka
* @Date:   2016-04-29 09:29:24
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-29 09:45:04
*/

'use strict';

class Carousel {

    // Constructor
    constructor(cardiv, thumbs) {
        this.carouseldiv = $(cardiv);
        this.thumbsdiv = (thumbs !== undefined) ? $(thumbs) : $('[id^=carousel-selector-]');

        // init the carousel
        this.carouseldiv.carousel({
            interval: 5000
        });

        // Event handlers
        this.thumbsdiv.on('click', this, this.handleThumbs);
        this.carouseldiv.on('slid.bs.carousel', this, this.updateText);
    }

    // Print
    print() {
        console.log('I am Carousel!');
    }

    // Handle the carousel thumbnails
    handleThumbs(event) {
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
    updateText(event) {
        var _this = event.data;
        var id = $('.item.active').data('slide-number');
        $('#carousel-text').html($('#slide-content-'+id).html());
    }

}


