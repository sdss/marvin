/*
* @Author: Brian Cherinka
* @Date:   2016-04-12 01:41:18
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-03-31 16:48:35
*/

module.exports = function(grunt) {

  require('load-grunt-tasks')(grunt);

  // Project configuration.
  grunt.initConfig({
    pkg: grunt.file.readJSON('package.json'),
    // Babel - transpiler from ES6 to ES5
    babel: {
        options: {
            sourceMap: false,
            presets: ['es2015']
        },
        dist: {
          files: [{
            expand: true,
            cwd: 'src',
            src: ['**/*.js'],
            dest: 'js',
            ext: '.js'
          }]
        }
    },
    // Bootstrap linter
    bootlint: {
      options: {
        relaxerror: [],
        showallerrors: false,
        stoponerror: false,
        stoponwarning: false
      },
      files: ['../templates/*.html']
    } ,
    // SASS to CSS
    sass: {
      dist: {
        files: [{
          expand: true,
          cwd: 'scss',
          src: ['**/*.scss'],
          dest: 'css',
          ext: '.css'
        }]
      }
    },
    // File Concatenation
    concat: {
        js: {
          options: {
              separator: ';'
          },
          src: ['js/*.js', '!js/{test,js9,wcs}*.js'],
          dest: 'dist/<%= pkg.name %>.js'
        },
        css: {
          src: ['css/*.css', '!css/js9*.css'],
          dest: 'dist/<%= pkg.name %>.css'
        }
    },
    // CSS Minification
    cssmin: {
      options: {
        sourceMap: true,
      },
      dist: {
        src: 'css/<%= pkg.name %>.css',
        dest: 'dist/<%= pkg.name %>.min.css'
      }
    },
    // JS Minification
    uglify: {
      options: {
        banner: '/*! <%= pkg.name %> <%= grunt.template.today("yyyy-mm-dd") %> */\n',
        compress: true
      },
      build: {
        src: 'dist/<%= pkg.name %>.js',
        dest: 'dist/<%= pkg.name %>.min.js'
      }
    },
    // Image Minification
    imagemin: {
      dynamic: {
        files: [{
          expand: true,
          cwd:'images',
          src: ['**/*.{png,jpg,jpeg,gif}'],
          dest: '../static/images'
        }]
      }
    }
  });

  // Load individual plugins that provide the tasks.  Commented out but left as an example.
  // Alternatively you can replace all tasks loads with the single line at the top require('load-grunt-tasks')(grunt);
  //grunt.loadNpmTasks('grunt-contrib-uglify');

  // Set default file path
  //grunt.file.setBase('../python/marvin/web/static/');

  // Default task(s). New tasks go in a tasklist.  Tasks are run in that order.
  grunt.registerTask('default', ['babel', 'sass', 'concat', 'newer:imagemin', 'newer:cssmin', 'newer:uglify']);
};
