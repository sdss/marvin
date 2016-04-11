#!/bin/bash

# @Author: Brian Cherinka
# @Date:   2016-04-11 16:40:19
# @Last Modified by:   Brian
# @Last Modified time: 2016-04-11 16:53:07

# Build script for Marvin
#
#
# Create files marvin.js, marvin.min.js, marvin.css, marvin.min.css
#
# Pre-requisite :
# - uglifyjs version 2 : https://github.com/mishoo/UglifyJS2
# - clean-css
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

# # css
# csssrcfile=${scriptdir}/marvin.css
# cssfileminified=${scriptdir}/marvin.min.css

# cssfiles=()
# cssfilelist=${cssfiles[@]}

# # single css file
# cat $cssfilelist > $csssrcfile
# # minified css
# cleancss -o $cssfileminified $csssrcfile

# js

jsfiles=('js/utils.js' 'js/olmap.js' 'js/galaxy.js')

# non-minified version
cat $filelist > $distribfile
# minified version
filelist=${jsfiles[@]}
uglifyjs $filelist --comments -c -o $distribfileminified

