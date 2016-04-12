/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 01:41:18
* @Last Modified by:   Brian
* @Last Modified time: 2016-04-12 11:33:28
*/

module.exports = function(grunt) {

  require('load-grunt-tasks')(grunt);

  // Project configuration.
  grunt.initConfig({
    pkg: grunt.file.readJSON('package.json'),
    // Babel - transpiler from ES6 to ES5
    babel: {
        options: {
            sourceMap: true,
            presets: ['babel-preset-es2015']
        },
        dist: {
            files: [{
                expand: true,
                '<%= pkg.name %>.js': '<%= pkg.name %>.js'
            }]
        }
    },
    // File Concatenation
    concat: {
        js: {
          options: {
              separator: ';'
          },
          dist: {
              src: ['js/*.js', '!js/test*.js', '!js/js9*.js', '!js/wcs*.js'],
              dest: '<%= pkg.name %>.js'
          }
        },
        css: {
          src: ['css/*.css', '!css/js9*.css'],
          dest: '<%= pkg.name %>.css'
        }
    },
    // CSS Minification
    cssmin: {
      dist: {
        src: '<%= pkg.name %>.css',
        dest: '<%= pkg.name %>.min.css'
      }
    },
    // JS Minification
    uglify: {
      options: {
        banner: '/*! <%= pkg.name %> <%= grunt.template.today("yyyy-mm-dd") %> */\n',
        compress: true
      },
      build: {
        src: '<%= pkg.name %>.js',
        dest: '<%= pkg.name %>.min.js'
      }
    }
  });

  // Load the plugins that provides the tasks
  //grunt.loadNpmTasks('grunt-babel');
  grunt.loadNpmTasks('grunt-contrib-concat');
  grunt.loadNpmTasks('grunt-contrib-cssmin');
  grunt.loadNpmTasks('grunt-contrib-uglify');

  // Set default file path
  grunt.file.setBase('../python/marvin/web/static/');

  // Default task(s).
  //grunt.registerTask('default', ['babel']);
  grunt.registerTask('default', ['concat', 'cssmin', 'uglify']);
};
