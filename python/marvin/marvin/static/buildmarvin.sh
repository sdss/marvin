#!/bin/bash

# Build script for Marvin
#
#
# Create files marvin.js, marvin.min.js, marvin.min.css
# 
# Pre-requisite : 
# - uglifyjs version 2 : https://github.com/mishoo/UglifyJS2
# - lessc
#
#

uglifyjs="/usr/local/bin/uglifyjs"
lessc="/usr/local/bin/lessc"

#scriptdir="$( cd "$( dirname "$0" )" && pwd )"
scriptdir=$(pwd)
echo $scriptdir
jssrcdir=${scriptdir}/js
#csssrcdir=${scriptdir}/css

distribfile=${scriptdir}/marvin.js
distribfileminified=${scriptdir}/marvin.min.js

sourcemap=${scriptdir}/marvin.map

# css
#csssrcfile=${srcdir}/css/marvin.css
#cssfiles=('index.css','tabs.css','verticals.css','vote.css', 'qunit-1.19.0.css')
#cssfileminified=${scriptdir}/marvin.min.css

# js

#jsfiles=('cds.js' 'json2.js' 'Logger.js' 'jquery.mousewheel.js' 'RequestAnimationFrame.js' 'Stats.js' 'healpix.min.js' 'astroMath.js' 'projection.js' 'coo.js' 'fits.js' 'CooConversion.js' 'Sesame.js' 'HealpixCache.js' 'Utils.js' 'URLBuilder.js' 'MeasurementTable.js' 'Color.js' 'AladinUtils.js' 'ProjectionEnum.js' 'CooFrameEnum.js' 'Downloader.js' 'CooGrid.js' 'Footprint.js' 'Popup.js' 'Circle.js' 'Polyline.js' 'Overlay.js' 'Source.js' 'ProgressiveCat.js' 'Catalog.js' 'Tile.js' 'TileBuffer.js' 'ColorMap.js' 'HpxImageSurvey.js' 'HealpixGrid.js' 'Location.js' 'View.js' 'Aladin.js')

#cmd="cat "
#for t in "${jsfiles[@]}"
#do
#    cmd="${cmd} ${srcdir}/js/$t"
#done

jsfiles=('js/jquery/jquery-1.11.2.min.js' 'js/jquery/jquery.actual.min.js' 'js/bootstrap/bootstrap.min.js' 'js/bootstrap/bootstrap-select.js' 
    'js/bootstrap/bootstrap-table-all.min.js' 'js/bootstrap/bootstrap-table-flat-json.js' 'js/jquery/moment.min.js' 
    'js/bootstrap/bootstrap-datetimepicker.min.js' 'js/bootstrap/bootstrap-tags.min.js' 'js/bootstrap/bloodhound.js' 
    'js/bootstrap/bootstrap3-typeahead.min.js' 'js/jquery/validator.min.js' 'js/d3/d3.min.js' 
    'js/qunit/qunit-1.19.0.js' 'js/qunit/qunit-assert-html.min.js' 'js/qunit/qunit-assert-classes.js' 'js/qunit/qunit-once.js' 
    'js/tableMethods.js' 'js/utils.js' 'js/aladin.min.js' 'js/header.js' 'js/search.js' 'js/feedback.js') 

#'js/dapqa.js' 'js/comments.js' 'js/ifu.js' 'js/plateinfo.js')

# non-minified version 
#cmd1="${cmd}  > ${distribfile}"
#eval ${cmd1}

# minified version
filelist=${jsfiles[@]}
uglifyjs $filelist --comments -c -o $distribfileminified
#fileList=""
#for t in "${jsfiles[@]}"
#do
#    fileList="${fileList} ${srcdir}/js/$t"
#done
#cmd2="${uglifyjs} ${fileList} --comments -c -m > ${distribfileminified}"
#eval ${cmd2}

# traitement des CSS
#${lessc} ${csssrcfile} ${cssfileminified}

