
.. _marvin-results_plot:

Plotting Your Results
---------------------

Once you perform a query and return some results, you may quickly plot and explore the returned parameters.  The Marvin `Results` object provides two
convenience functions for quickly creating a 2-d scatter plot, or a histogram on a single column.  All plotting routines in `Results` will, by default, grab **all** data for the requested columns before plotting, even if you only have a subset of the data loaded locally in a `ResultSet`.

Creating a scatter plot
^^^^^^^^^^^^^^^^^^^^^^^

Use the Results :meth:`plot <marvin.tools.results.Results.plot>` method to quickly create a 2-d scatter plot using Matplotlib.  The Results `plot` method is a thinly-wrapped version of the :func:`marvin.utils.plot.scatter.plot` utility function.  See :ref:`marvin-utils-plot-scatter` for full details on how to use this function.

::

    # create a scatter plot of redshift vs g-r color, from a Results
    r.plot('z', 'absmag_g_r')

By default, `plot` will also include the marginal distributions for the x and y parameters.  You can turn off the histogram options in the scatter plot with `with_hist=False`, or specify a single axis with `with_hist='x'`.

Creating a histogram
^^^^^^^^^^^^^^^^^^^^

Use the Results `hist` method to quickly create a 1-d histogram using Matplotlib.  The Results `hist` method is a thinly-wrapped version of the :func:`marvin.utils.plot.scatter.hist` utility function.  See :ref:`marvin-utils-plot-hist` for full details on how to use this function.

::

    # create a histogram of redshift from a Results
    r.hist('z')


Interacting with Histogram Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both the `hist` method, and the `plot` method (when also returning marginal distributions) will include an additional key,value pair inside the standard histogram output (see :ref:`marvin-utils-plot-hist-using`).  This additional key,value pair is called **bins_plateifu**, and is a dictionary of binids, each containing a list of targets (**plateifus**) within that bin, for easy target identification.

Let's create a scatter plot of redshift vs absolute magnitude g-r color, with the respective x and y histograms::

    output = r.plot('z', 'absmag_g_r')

This returns a tuple containing the (matplotlib Figure, the matplotlib Axes objects, the histogram output).  The histogram data is a dictionary, with each axis indicated as **xhist** and **yhist** keys, respectively.  Let's extract the histogram data for redshift data.::

    # get the histogram data
    hdata = output[3]

    # redshift was plotted on the x-axis, so lies in the xhist key
    zdata = hdata['xhist']

Let's get the list of targets in the 2nd redshift bin.  Note the key syntax for accessing `bins_plateifu` is **key:binid**, and this is 1-indexed.::

    # 2nd redshift bin range
    print('{0} to {1}'.format(zdata['binedges'][1], zdata['binedges'][2]))
    0.00654206261551 to 0.012540415231

    # get the list of targets in the 2nd redshift bin
    bin3_targets = zdata['bins_plateifu'][2]
    [u'7990-12703', u'8135-6101', u'8332-1902']

Let's get the redshift and g-r colors for the targets in the 2nd bin.  We can use the :func:`marvin.utils.general.map_bins_to_columns` utility function to map any array data into the specified bin groups::

    from marvin.utils.general import map_bins_to_column

    # get the redshift and color data
    redshift = r.getListOf('z', return_all=True)
    g_r = r.getListOf('absmag_g_r', return_all=True)

    # map the redshift and g-r color array data into our redshift bins
    redshift_bins = map_bins_to_columns(redshift, zdata['indices'])
    g_r_bins = map_bins_to_columns(g_r, zdata['indices'])

    # print the data in the 2nd redshift bin
    print(redshift_bins[2])
    [0.0113986, 0.0108501, 0.00814191]

    print(g_r_bins[2])
    [1.00057220458984, 0.984879493713379, 1.25316619873047]

    print(zip(zdata['bins_plateifu'][2], red_bins[2], gr_bins[2]))
    [(u'7990-12703', 0.0113986, 1.00057220458984),
     (u'8135-6101', 0.0108501, 0.984879493713379),
     (u'8332-1902', 0.00814191, 1.25316619873047)]



