#!/usr/bin/env python
# encoding: utf-8
#
# test_util.py
#
# Created by José Sánchez-Gallego on 5 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from matplotlib import pyplot as plt
import numpy as np

from marvin.utils.plot import bind_to_figure


class TestBindToFigure(object):

    def test_bind_to_figure(self):

        # Creates some axes to copy
        base_fig, ax = plt.subplots()

        # Adds plots, texts, scatter, and imshows
        ax.plot([1, 2], [1, 2], 'k-', label='Plot1')
        ax.plot([2, 3], [3, 4], 'r--', label='Plot2')

        ax.scatter([1, 2, 3, 4], [5, 6, 7, 8], s=10, marker='s')

        ax.text(1, 2, 'Hola', fontsize=12)

        ones_sample = np.ones((10, 10, 3), dtype=np.uint8)
        ax.imshow(ones_sample)

        ax.legend()

        # Binds the axes to a new image and tests the copy
        new_fig = bind_to_figure(ax)

        custom_fig, __ = plt.subplots()

        custom_fig_bound = bind_to_figure(ax, fig=custom_fig)

        for fig in [new_fig, custom_fig_bound]:

            assert len(fig.axes) == 1

            new_axes = fig.axes[0]

            assert len(new_axes.lines) == 2
            assert np.all(new_axes.lines[-1].get_data()[1] == [3, 4])
            assert new_axes.lines[-1].get_linestyle() == '--'
            assert new_axes.lines[-1].get_color() == 'r'
            assert new_axes.lines[-1].get_label() == 'Plot2'

            assert len(new_axes.collections) == 1
            assert np.all(new_axes.collections[0].get_offsets()[:, 1] == [5, 6, 7, 8])
            assert np.all(new_axes.collections[0].get_sizes() == 10)

            assert len(new_axes.texts) == 1
            assert new_axes.texts[0].get_text() == 'Hola'
            assert new_axes.texts[0].get_fontsize() == 12
            assert np.all(new_axes.texts[0].get_position() == (1, 2))

            assert len(new_axes.images) == 1
            assert np.all(new_axes.images[0].get_array() == ones_sample)

            assert new_axes.legend_ is not None
