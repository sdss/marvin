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

scriptdir="$( cd "$( dirname "$0" )" && pwd )"
jssrcdir=${scriptdir}/js
csssrcdir=${scriptdir}/css


distribfile=${scriptdir}/marvin.js
distribfileminified=${scriptdir}/marvin.min.js

# css
csssrcfile=${srcdir}/css/marvin.css
cssfiles=('index.css','tabs.css','verticals.css','vote.css', 'qunit-1.19.0.css')
cssfileminified=${scriptdir}/marvin.min.css

cmd="cat "
for t in "${cssfiles[@]}"
do
    cmd="${cmd} ${srcdir}/css/$t"
done

# js

jsfiles=('cds.js' 'json2.js' 'Logger.js' 'jquery.mousewheel.js' 'RequestAnimationFrame.js' 'Stats.js' 'healpix.min.js' 'astroMath.js' 'projection.js' 'coo.js' 'fits.js' 'CooConversion.js' 'Sesame.js' 'HealpixCache.js' 'Utils.js' 'URLBuilder.js' 'MeasurementTable.js' 'Color.js' 'AladinUtils.js' 'ProjectionEnum.js' 'CooFrameEnum.js' 'Downloader.js' 'CooGrid.js' 'Footprint.js' 'Popup.js' 'Circle.js' 'Polyline.js' 'Overlay.js' 'Source.js' 'ProgressiveCat.js' 'Catalog.js' 'Tile.js' 'TileBuffer.js' 'ColorMap.js' 'HpxImageSurvey.js' 'HealpixGrid.js' 'Location.js' 'View.js' 'Aladin.js')

cmd="cat "
for t in "${jsfiles[@]}"
do
    cmd="${cmd} ${srcdir}/js/$t"
done


# version non minifiée
cmd1="${cmd}  > ${distribfile}"
eval ${cmd1}

# version minifiée
fileList=""
for t in "${jsfiles[@]}"
do
    fileList="${fileList} ${srcdir}/js/$t"
done
cmd2="${uglifyjs} ${fileList} --comments -c -m > ${distribfileminified}"
eval ${cmd2}

# traitement des CSS
${lessc} ${csssrcfile} ${cssfileminified}

