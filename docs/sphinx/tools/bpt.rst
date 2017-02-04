.. _marvin-bpt:

BPT diagrams
============

A skillfully written general description of the BPT method will go here.

Marvin now includes the ability to generate BPT diagrams for a particular galaxy.  Marvin makes use of the classification system defined by `Kewley et al. (2006) <https://ui.adsabs.harvard.edu/#abs/2006MNRAS.372..961K/abstract>`_  to return classification masks for different ionisation mechanisms.  By default, the Marvin BPT uses a strict selection criteria, utilizing all three BPTs (NII, SII, and OI) from Kewley2006.  A spaxel only becomes classified if it meets the criteria in all three.

The BPT spaxel classifications that Marvin provides are

* **Star-Forming (sf)**:
    Spaxels that fall in the Kewley Star-forming region in the SII and OI BPTs, and the Kauffmann Star-forming region in the NII BPT.

* **Composite (comp)**:
    Spaxels that fall in the Kewley Star-forming region in the SII and OI BPTs, and between the Kauffmann Star-forming region and the Kewley Star-forming region in the NII BPT.

* **AGN (agn)**:
    Spaxels that fall in the AGN region in all three BPTs.

* **Seyfert (seyfert)**:
    Spaxels that fall in the AGN region in all three BPTs, and the Seyfert region in both the SII, and OI BPTs.

* **Liner (liner)**:
    Spaxels that fall in the AGN region in all three BPTs, and the Liner region in both the SII, and OI BPTs.

* **Ambiguous (ambiguous)**:
    Spaxels that cannot be strictly classified in one of the given categories.

* **Invalid (invalid)**:
    Spaxels that have emission-line flux < 0 and are rejected by any SNR cuts.

Often the OI diagnostic line cannot be realiably measured.  If you wish to disable the use of the OI diagnostic line when classifying your spaxels, use may set the ``use_oi`` keyword to ``False``.  This turns off the OI line, and only uses the NII, and SII BPTs during classification, giving you more spaxels to play with.

By default, :meth:`~marvin.tools.maps.Maps.get_bpt` produces and returns a matplotlib figure with the classification plots **(based on Kewley+06 Fig. 4)** and the 2D spatial distribution of classified spaxels (i.e., a map of the galaxy in which each spaxel is colour-coded based on its emission mechanism).  To disable the return of the figure, you may set the ``return_figure`` keyword to ``False``.

See :meth:`~marvin.tools.maps.Maps.get_bpt` for the API reference of how to call BPT from within Maps.  See :ref:`marvin-utils-bpt` for the API reference guide on the BPT utility code.

::

    # get a map
    maps = Maps(plateifu='8485-1901')

    # make a standard 3-plot BPT and retrieve the classifications
    masks, fig = maps.get_bpt()

    # make a BPT classification without the OI
    masks, fig = maps.get_bpt(use_oi=False)

    # I only want the masks (good for batch jobs)
    masks = maps.get_bpt(return_figure=False, show_plot=False)

    # Give me the masks, and figures but don't show me the plot (good for batch jobs)
    masks, fig = maps.get_bpt(show_plot=False)

Signal-To-Noise Cuts
--------------------

Marvin's BPT code allows you to impose a cut on SN over any or all of the emission-line diagnostics used in spaxel classification.  Marvin accepts either a single number, which will be applied to all emission-lines, or a dictionary of values for specific emission lines.  **Marvin uses a default SN cutoff value of 3.**

When using a dictionary to define your SN cutoffs, it takes the form of ``{emission_line: sn_threshold}``.  The emission lines available are
``ha``, ``hb``, ``nii``, ``sii``, ``oiii``, and ``oi``.  Any lines not specified in the dictionary take on the default value of 3.

::

    maps = Maps(plateifu='8485-1901')

    # generate a bpt plot using a single SNR cutoff of 5
    masks, fig = maps.get_bpt(snr=5)

    # generate a bpt plot using an Ha SN cutoff of 5 and a SII SN cutoff of 2.  The remaining lines are cutoff at SNR of 3.
    sndict = {'ha': 5, 'sii':2}
    masks, fig = maps.get_bpt(snr=sndict)

Using the Masks
---------------

Marvin always returns the BPT classifications as masks.  These masks are boolean arrays of the same shape as Maps, i.e. 2d-arrays. These masks can be used to filter on any other Map or Cube property.  Marvin returns a dictionary of all the classifications.

::

    maps = Maps(plateifu='8485-1901')

    # generate a bpt plot and retrieve the masks
    masks, fig = maps.get_bpt()

    # look at the masks included in this dictionary
    print(masks.keys())
    ['agn', 'ambiguous', 'comp', 'liner', 'invalid', 'seyfert', 'sf']

    # each mask is a boolean 2-d array of the same shape as the Maps
    print(masks['sf'])
    array([[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]], dtype=bool)

    print(masks['sf'].shape)
    (34, 34)

    # let's look at the H-alpha EW values for all spaxels classified as Star-Forming (sf)

    # get the Ha EW map
    haew = maps.getMap('emline_sew', channel='ha_6564')
    print(haew)
    <Marvin Map (plateifu='8485-1901', property='emline_sew', channel='ha_6564')>

    # select and view the ew for star-forming spaxels
    sfewha = haew.value[masks['sf']]
    print(sfewha)
    array([ 26.52018748,  28.51509129,  29.21568103,  29.02369049,
            26.76387933,  28.51799067,  28.88143649,  28.33309614,
            28.05468761,  27.37624124,  24.37387385,  26.04795531,
            27.4333648 ,  27.67205947,  27.1107335 ,  26.73307361,
            24.24404652,  25.0204489 ,  26.0995353 ,  26.79414024,
            26.63586029,  25.87115022,  25.70280123,  27.16742326,
            28.05049556,  27.81402451,  26.3372375 ,  29.53938287])

If you want to know the spaxel x, y coordinates for the spaxels in given mask, you can use `Numpy's where <https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html>`_ command.  Using Numpy's where on a boolean array will return the indices of that array that evaluate to ``True``. **Note that for 2d-arrays, numpy.where always returns a tuple of (array of y indices, array of x indices).**

::

    # get a mask
    masks, fig = maps.get_bpt()

    # get the spaxel x, y coordinates of our star-forming spaxels
    import numpy as np
    y, x = np.where(masks['sf'])
    y
    array([13, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19,
            19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 21])
    x
    array([16, 16, 17, 18, 15, 16, 17, 18, 19, 20, 14, 15, 16, 17, 18, 19, 14,
            15, 16, 17, 18, 19, 20, 16, 17, 18, 19, 18])

    # using the star-forming HaEW selection from before
    print(sfewha)
    array([ 26.52018748,  28.51509129,  29.21568103,  29.02369049,
            26.76387933,  28.51799067,  28.88143649,  28.33309614,
            28.05468761,  27.37624124,  24.37387385,  26.04795531,
            27.4333648 ,  27.67205947,  27.1107335 ,  26.73307361,
            24.24404652,  25.0204489 ,  26.0995353 ,  26.79414024,
            26.63586029,  25.87115022,  25.70280123,  27.16742326,
            28.05049556,  27.81402451,  26.3372375 ,  29.53938287])

    # Let's verify this, by looking at the individual spaxel values
    # Since numpy.where returns Numpy 0-based indices, we select spaxels using bracket notation [x, y]

    # let's check the first one y=13, x=16.
    spaxel = maps[x[0], y[0]]
    spaxel.properties['emline_sew_ha_6564']
    <AnalysisProperty (name=emline_sew, channels=ha_6564, value=26.5201874768 ivar=26.8136299521, mask=0)>

    # the value property matches the first element in our sfewha array.
    # Let's check the 2nd one at y=16, x=16
    spaxel = maps[x[1], y[1]]
    <AnalysisProperty (name=emline_sew, channels=ha_6564, value=28.5150912875 ivar=76.8864418103, mask=0)>

    # It matches!

Modifying the Plot
------------------

Once you return the BPT figure, you are free to modify it anyway you like.



