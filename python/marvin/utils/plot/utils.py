#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under a 3-clause BSD license.
#
# Original code from mangadap.plot.colorbar.py licensed under the following
# 3-clause BSD license.
#
# Copyright (c) 2015, SDSS-IV/MaNGA Pipeline Group
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
# HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# utils.py
#
# Created by José Sánchez-Gallego on 5 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from matplotlib import pyplot as plt


__ALL__ = ('bind_to_figure')


def bind_to_figure(ax, fig=None):
    """Copies axes to a new figure.

    This is a custom implementation of a method to copy axes from one
    matplotlib figure to another. Matplotlib does not allow this, so we create
    a new figure and copy the relevant lines and containers.

    This is all quite hacky and may stop working in future versions of
    matplotlib, but it seems to be the only way to bind axes from one figure
    to a different one.

    Current limitations include: 1) the legend is copied but its style is not
    maintained; 2) scatter plots do not maintain the marker type, markers are
    always replaced with squares.

    """

    if fig is not None:
        assert isinstance(fig, plt.Figure), 'argument must be a Figure'
        assert len(fig.axes) == 1, 'figure must have one and only one axes'
        new_ax = fig.axes[0]
    else:
        fig, new_ax = plt.subplots()

    new_ax.set_facecolor(ax.get_facecolor())

    for line in ax.lines:
        data = line.get_data()
        new_ax.plot(data[0], data[1], linestyle=line.get_linestyle(), color=line.get_color(),
                    zorder=line.zorder, label=line.get_label())

    for collection in ax.collections:
        data = collection.get_offsets()
        new_ax.scatter(data[:, 0], data[:, 1], marker='s', facecolor=collection.get_facecolors(),
                       edgecolor=collection.get_edgecolors(), s=collection.get_sizes(),
                       zorder=line.zorder, label=collection.get_label())

    for text in ax.texts:
        xx, yy = text.get_position()
        new_ax.text(xx, yy, text.get_text(), family=text.get_fontfamily(),
                    fontsize=text.get_fontsize(),
                    color=text.get_color(), ha=text.get_horizontalalignment(),
                    va=text.get_verticalalignment(), zorder=text.zorder)

    for image in ax.images:
        new_ax.imshow(image.get_array(), interpolation=image.get_interpolation())

    if ax.legend_:
        new_ax.legend()

    new_ax.grid(ax.get_xgridlines(), color=ax.get_xgridlines()[0].get_color(),
                alpha=ax.get_xgridlines()[0].get_alpha())
    new_ax.grid(ax.get_ygridlines(), color=ax.get_xgridlines()[0].get_color(),
                alpha=ax.get_xgridlines()[0].get_alpha())

    new_ax.set_xlim(ax.get_xlim())
    new_ax.set_ylim(ax.get_ylim())
    new_ax.set_xlabel(ax.get_xlabel())
    new_ax.set_ylabel(ax.get_ylabel())

    return fig
