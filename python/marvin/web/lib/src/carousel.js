/*
* @Author: Brian Cherinka
* @Date:   2016-04-29 09:29:24
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:11:43
*/

//jshint esversion: 6
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
        let _this = event.data;
        let id_selector = $(this).attr("id");
        try {
            let id = /-(\d+)$/.exec(id_selector)[1];
            //console.log(id_selector, id);
            _this.carouseldiv.carousel(parseInt(id));
        } catch (e) {
            console.log('MyCarousel: Regex failed!', e);
        }
    }

    // When carousel slides, auto update the text
    updateText(event) {
        let _this = event.data;
        let id = $('.item.active').data('slide-number');
        $('#carousel-text').html($('#slide-content-'+id).html());
    }

}


