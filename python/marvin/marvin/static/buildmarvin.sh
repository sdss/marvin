#!/bin/bash

# Build script for Marvin
#
#
# Create files marvin.js, marvin.min.js, marvin.css, marvin.min.css
# 
# Pre-requisite : 
# - uglifyjs version 2 : https://github.com/mishoo/UglifyJS2
# - clean-css
#
#

uglifyjs="/usr/local/bin/uglifyjs"
cleancss="/usr/local/bin/cleancss"

scriptdir=$(pwd)
echo $scriptdir
jssrcdir=${scriptdir}/js
csssrcdir=${scriptdir}/css

distribfile=${scriptdir}/marvin.js
distribfileminified=${scriptdir}/marvin.min.js

sourcemap=${scriptdir}/marvin.map

# css
csssrcfile=${scriptdir}/marvin.css
cssfileminified=${scriptdir}/marvin.min.css

cssfiles=('css/bootstrap/bootstrap.min.css' 'css/bootstrap/bootstrap-select.css' 'css/bootstrap/bootstrap-table.css' 
    'css/bootstrap/bootstrap-datetimepicker.min.css' 'css/bootstrap/bootstrap-tags.css' 'css/index.css' 'css/verticals.css' 
    'css/vote.css' 'css/tabs.css' 'css/qunit-1.19.0.css' 'css/aladin.min.css')
cssfilelist=${cssfiles[@]}

# single css file
cat $cssfilelist > $csssrcfile
# minified css
cleancss -o $cssfileminified $csssrcfile 

# js

jsfiles=('js/jquery/jquery-1.11.2.min.js' 'js/jquery/jquery.actual.min.js' 'js/bootstrap/bootstrap.min.js' 'js/bootstrap/bootstrap-select.js' 
    'js/bootstrap/bootstrap-table-all.min.js' 'js/bootstrap/bootstrap-table-flat-json.js' 'js/jquery/moment.min.js' 
    'js/bootstrap/bootstrap-datetimepicker.min.js' 'js/bootstrap/bootstrap-tags.min.js' 'js/bootstrap/bloodhound.js' 
    'js/bootstrap/bootstrap3-typeahead.min.js' 'js/jquery/validator.min.js' 'js/d3/d3.min.js' 
    'js/qunit/qunit-1.19.0.js' 'js/qunit/qunit-assert-html.min.js' 'js/qunit/qunit-assert-classes.js' 'js/qunit/qunit-once.js' 
    'js/tableMethods.js' 'js/utils.js' 'js/aladin.min.js' 'js/header.js' 'js/search.js' 'js/feedback.js') 

#'js/dapqa.js' 'js/comments.js' 'js/ifu.js' 'js/plateinfo.js')

# non-minified version 
cat $filelist > $distribfile
# minified version
filelist=${jsfiles[@]}
uglifyjs $filelist --comments -c -o $distribfileminified

