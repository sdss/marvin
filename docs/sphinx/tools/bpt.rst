.. _marvin-bpt:

BPT Diagrams
------------

Marvin now includes the ability to generate BPT (`Baldwin, Phillips, & Terlevich 1981 <https://ui.adsabs.harvard.edu/#abs/1981PASP...93....5B/abstract>`_) diagrams for a particular galaxy.  Marvin makes use of the classification system defined by |kewley2006|_  to return classification masks for different ionisation mechanisms.  By default, the Marvin BPT uses a strict selection criteria, utilizing all three BPT diagnostic criteria (NII, SII, and OI) from |kewley2006|_.  A spaxel only becomes classified if it meets the criteria in all three.

The BPT spaxel classifications that Marvin provides are

* **Star-Forming (sf)**:
    Spaxels that fall in the |kewley2006|_ star-forming region in the SII and OI BPTs, and the |kauffmann2003|_ star-forming region in the NII BPT.

* **Composite (comp)**:
    Spaxels that fall in the |kewley2006|_ star-forming region in the SII and OI BPTs, and between the |kauffmann2003|_ star-forming region and the |kewley2006|_ star-forming region in the NII BPT.

* **AGN (agn)**:
    Spaxels that fall in the AGN region in all three BPTs.

* **Seyfert (seyfert)**:
    Spaxels that fall in the AGN region in all three BPTs, and the Seyfert region in both the SII, and OI BPTs.

* **LINER (liner)**:
    Spaxels that fall in the AGN region in all three BPTs, and the LINER region in both the SII, and OI BPTs.

* **Ambiguous (ambiguous)**:
    Spaxels that cannot be strictly classified in one of the given categories.

* **Invalid (invalid)**:
    Spaxels that have emission line flux <= 0 or are below the minimum signal-to-noise ratio (SNR).

Often the OI diagnostic line cannot be reliably measured.  If you wish to disable the use of the OI diagnostic line when classifying your spaxels, use may set the ``use_oi`` keyword to ``False``.  This turns off the OI line, and only uses the NII and SII BPTs during classification, giving you more spaxels to play with.

By default, :meth:`~marvin.tools.maps.Maps.get_bpt` produces and returns a matplotlib figure with the classification plots **(based on |kewley2006|_ Fig. 4)**, a list of matplotlib axes for each subplot, and the 2D spatial distribution of classified spaxels (i.e., a map of the galaxy in which each spaxel is colour-coded based on its emission mechanism).  To disable the return of the figure and axes, set the ``return_figure`` keyword to ``False``.

See :meth:`~marvin.tools.maps.Maps.get_bpt` for the API reference of how to generate a BPT diagram from within :ref:`marvin-tools-maps`.  See :ref:`marvin-utils-bpt` for the API reference guide on the BPT utility code.

.. plot::
    :align: center
    :include-source: True

    # get a map
    from marvin.tools import Maps
    maps = Maps('8485-1901')

    # make a standard 3-plot BPT and retrieve the classifications
    masks, fig, axes = maps.get_bpt()

    # save the plot
    # fig.savefig('bpt.png')

    # make a BPT classification without OI
    masks, fig, axes = maps.get_bpt(use_oi=False)


.. code-block:: python

    # also show the optical image
    from marvin.utils.general.images import showImage
    image = showImage(plateifu='8485-1901')

    # only return the masks (good for batch jobs)
    masks = maps.get_bpt(return_figure=False, show_plot=False)

    # give me the masks and figures, but don't show me the plot (good for batch jobs)
    masks, fig, axes = maps.get_bpt(show_plot=False)


Minimum Signal-To-Noise Ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marvin's BPT code allows you to impose a minimum SNR over any or all of the emission line diagnostics used in spaxel classification.  Marvin accepts either a single number, which will be applied to all emission lines, or a dictionary of values for specific emission lines.  **Marvin uses a default minimum SNR of 3.**

When using a dictionary to define your minimum SNR, it takes the form of ``{emission_line: snr_min}``.  The emission lines available are ``ha``, ``hb``, ``nii``, ``sii``, ``oiii``, and ``oi``.  Any lines not specified in the dictionary take on the default value of 3.

.. code-block:: python

    maps = Maps(plateifu='8485-1901')

    # generate a bpt plot using a sinlge minimum SNR of 5
    masks, fig, axes = maps.get_bpt(snr_min=5)

    # generate a bpt plot using a minimum Halpha SNR of 5 and a minimum SII SNR of 2.  The remaining lines have minimum SNRs of 3.
    snrdict = {'ha': 5, 'sii': 2}
    masks, fig, axes = maps.get_bpt(snr_min=snrdict)


Using the Masks
^^^^^^^^^^^^^^^

Marvin always returns the BPT classifications as masks.  These masks are boolean arrays of the same shape as :ref:`marvin-tools-maps`, i.e. 2d-arrays. These masks can be used to filter on any other :ref:`marvin-tools-map` or :ref:`marvin-tools-cube` property.  Marvin returns a dictionary of all the classifications, with two tiers.  At the top level, the BPT mask contains a key for each classfication category.  Within each category, there are four sub-groups, described as follows:

* **global**: the strict spaxel classifications as described above, using all three BPT diagrams
* **nii**: the spaxel classifications using only the NII BPT
* **sii**: the spaxel classifications using only the SII BPT
* **oi**: the spaxel classifications using only the OI BPT

.. code-block:: python

    >>> maps = Maps(plateifu='8485-1901')

    # generate a bpt plot and retrieve the masks
    >>> masks, fig, axes = maps.get_bpt()

    # look at the masks included in this dictionary
    >>> print(masks.keys())
    dict_keys(['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous'])

    # each mask is a boolean 2-d array of the same shape as the Maps
    >>> masks['sf']['global']
    array([[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]], dtype=bool)

    >>> print(masks['sf']['global'].shape)
    (34, 34)

    # let's look at the H-alpha EW values for all spaxels classified as star-Forming (sf)

    # get the Ha EW map
    >>> haew = maps.getMap('emline_sew', channel='ha_6564')
    >>> haew
    <Marvin Map (property='emline_sew_ha_6564')>
    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ...,
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]] Angstrom

    # select and view the ew for star-forming spaxels
    >>> sfewha = haew.value[masks['sf']['global']]
    >>> sfewha
    array([ 23.04647827,  22.36664963,  23.70358849,  23.62845421,
            24.51483345,  25.4575119 ,  25.2571373 ,  24.0802269 ,
            22.67666435,  19.39162827,  16.50460052,  23.33211136,
            25.80060196,  25.62438965,  26.62814331,  27.80005455,
            28.19480896,  27.24209976,  25.19938469,  23.2147274 ,
            19.58403015,  14.56358242,  17.57133484,  23.82813644,
            26.2010479 ,  26.28424072,  28.06950569,  28.97672653,
            29.12378502,  28.88417625,  27.72723007,  24.07551575,
            20.87368774,  15.92866325,  18.56455231,  20.44847298,
            22.9385128 ,  25.85798645,  28.22526932,  29.16204071,
            29.5326519 ,  29.43461227,  28.35850143,  24.97596359,
            20.42848015,  15.66413593,  19.34163094,  21.91408539,
            26.08240509,  28.54499054,  29.47539902,  29.13975906,
            29.01648331,  28.41638374,  25.63819122,  21.42501068,
            20.37047958,  23.30433655,  26.76013374,  28.56043434,
            28.79559326,  28.40997696,  28.30820465,  27.90911293,
            26.18356323,  23.10487366,  22.8608532 ,  24.19278717,
            26.12378693,  27.61821365,  27.78279114,  27.38418961,
            27.13437271,  26.80350304,  26.20197105,  23.82313919,
            19.44246101,  23.36117363,  24.05638313,  25.21157074,
            26.43170166,  27.0764122 ,  26.98272896,  26.35611916,
            26.1333828 ,  25.82810402,  20.33587646,  23.84975243,
            24.93754196,  26.24217987,  27.01878929,  28.10024452,
            27.75396538,  26.75156212,  26.40979004,  26.73135185,
            28.82616615,  29.7464962 ,  30.21625328,  29.3112545 ,
            27.70197487,  26.9072876 ,  24.94372368,  30.46117592,
            30.43259811,  29.84792709,  29.16290665,  28.12854195,
            26.50462914,  24.89401054,  21.67862701,  30.13232803,
            28.73386765,  28.2321434 ,  27.89228249,  25.92523384,
            23.35713577,  17.73891258,  29.29098129,  28.42762566,
            28.28386498,  27.35419083,  23.70591164,  20.17571831,
            29.04303551,  29.63247871,  27.78384781,  24.58441162])

If you want to know the spaxel x, y coordinates for the spaxels in given mask, you can use Numpy's `np.where <https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html>`_ function.  Using ``np.where`` on a boolean array will return the indices of that array that evaluate to ``True``. **Note that for Maps, np.where returns a tuple of (array of y indices, array of x indices). Numpy stores data in row major ordered**

.. code-block:: python

    # get a mask
    >>> masks, fig, axes = maps.get_bpt()

    # get the spaxel x, y coordinates of our star-forming spaxels
    >>> import numpy as np
    >>> y, x = np.where(masks['sf']['global'])
    >>> print(y)
    [11 11 12 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13 13 13 14 14 14 14
     14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15 15 15 15 16 16 16 16 16
     16 16 16 16 16 16 17 17 17 17 17 17 17 17 17 17 18 18 18 18 18 18 18 18 18
     18 19 19 19 19 19 19 19 19 19 19 20 20 20 20 20 20 20 20 20 21 21 21 21 21
     21 21 21 22 22 22 22 22 22 22 22 23 23 23 23 23 23 23 24 24 24 24 24 24 25
     25 25 25]
    >>> print(x)
    [17 18 14 15 16 17 18 19 20 21 22 13 14 15 16 17 18 19 20 21 22 11 12 13 14
     15 16 17 18 19 20 21 22 11 12 13 14 15 16 17 18 19 20 21 22 11 13 14 15 16
     17 18 19 20 21 22 13 14 15 16 17 18 19 20 21 22 13 14 15 16 17 18 19 20 21
     22 12 13 14 15 16 17 18 19 20 21 12 13 14 15 16 17 18 19 20 14 15 16 17 18
     19 20 21 15 16 17 18 19 20 21 22 16 17 18 19 20 21 22 16 17 18 19 20 21 16
     17 18 19]

    # alternatively, if you want a list of coordinate pairs of [y, x]
    >>> coordlist = np.asarray(np.where(masks['sf']['global'])).T.tolist()
    >>> print(coordlist[0:2])
    [[11, 17], [11, 18]]

    # using the star-forming HaEW selection from before
    >>> print(sfewha)
    array([ 23.04647827,  22.36664963,  23.70358849,  23.62845421,
            24.51483345,  25.4575119 ,  25.2571373 ,  24.0802269 ,
            22.67666435,  19.39162827,  16.50460052,  23.33211136,
            25.80060196,  25.62438965,  26.62814331,  27.80005455,
            28.19480896,  27.24209976,  25.19938469,  23.2147274 ,
            19.58403015,  14.56358242,  17.57133484,  23.82813644,
            26.2010479 ,  26.28424072,  28.06950569,  28.97672653,
            29.12378502,  28.88417625,  27.72723007,  24.07551575,
            20.87368774,  15.92866325,  18.56455231,  20.44847298,
            22.9385128 ,  25.85798645,  28.22526932,  29.16204071,
            29.5326519 ,  29.43461227,  28.35850143,  24.97596359,
            20.42848015,  15.66413593,  19.34163094,  21.91408539,
            26.08240509,  28.54499054,  29.47539902,  29.13975906,
            29.01648331,  28.41638374,  25.63819122,  21.42501068,
            20.37047958,  23.30433655,  26.76013374,  28.56043434,
            28.79559326,  28.40997696,  28.30820465,  27.90911293,
            26.18356323,  23.10487366,  22.8608532 ,  24.19278717,
            26.12378693,  27.61821365,  27.78279114,  27.38418961,
            27.13437271,  26.80350304,  26.20197105,  23.82313919,
            19.44246101,  23.36117363,  24.05638313,  25.21157074,
            26.43170166,  27.0764122 ,  26.98272896,  26.35611916,
            26.1333828 ,  25.82810402,  20.33587646,  23.84975243,
            24.93754196,  26.24217987,  27.01878929,  28.10024452,
            27.75396538,  26.75156212,  26.40979004,  26.73135185,
            28.82616615,  29.7464962 ,  30.21625328,  29.3112545 ,
            27.70197487,  26.9072876 ,  24.94372368,  30.46117592,
            30.43259811,  29.84792709,  29.16290665,  28.12854195,
            26.50462914,  24.89401054,  21.67862701,  30.13232803,
            28.73386765,  28.2321434 ,  27.89228249,  25.92523384,
            23.35713577,  17.73891258,  29.29098129,  28.42762566,
            28.28386498,  27.35419083,  23.70591164,  20.17571831,
            29.04303551,  29.63247871,  27.78384781,  24.58441162])

    # Let's verify this, by looking at the individual spaxel values

    # let's check the first one y=11, x=17.
    >>> spaxel = maps[y[0], x[0]]
    >>> spaxel.emline_sew_ha_6564
    <AnalysisProperty 23.0464782715 Angstrom>

    # the value property matches the first element in our sfewha array.
    # Let's check the 2nd one at y=11, x=18
    >>> spaxel = maps[y[1], x[1]]
    <AnalysisProperty 22.3666496277 Angstrom>

    # It matches!

If you want to examine the emission-line ratios up close for spaxels in a given mask, you can do so easily using the rest of the Marvin :ref:`marvin-tools-maps`

.. code-block:: python

    # get a mask
    >>> masks, fig = maps.get_bpt()

    # get the nii_to_ha emission-line map
    >>> niihamap = maps['emline_gflux_nii_6585'] / maps['emline_gflux_ha_6564']

    # we need Numpy to take the log.  Let's look at the nii_to_ha values for the star-forming spaxels
    >>> import numpy as np
    >>> np.log10(niihamap.value)[masks['sf']['global']]
    array([-0.36083685, -0.35050373, -0.39707415, -0.38970575, -0.37744072,
           -0.37097652, -0.36574841, -0.36696256, -0.36225319, -0.33948732,
           -0.30500662, -0.40887598, -0.41479702, -0.39309623, -0.38104635,
           -0.38231165, -0.38451816, -0.38412328, -0.3857764 , -0.37604272,
           -0.34847514, -0.31711751, -0.35483825, -0.40735977, -0.40479966,
           -0.38222836, -0.38626824, -0.38634657, -0.38754567, -0.39089759,
           -0.3982011 , -0.39197492, -0.37097201, -0.33713065, -0.3631672 ,
           -0.34368277, -0.34618552, -0.36599119, -0.38414863, -0.39020342,
           -0.39107684, -0.39222271, -0.39853205, -0.40006774, -0.37997194,
           -0.31424691, -0.29854483, -0.32601963, -0.37445885, -0.39861996,
           -0.39477471, -0.39359296, -0.39470471, -0.39647526, -0.40258077,
           -0.39138844, -0.3133486 , -0.34971283, -0.39177572, -0.41063068,
           -0.40759568, -0.39895744, -0.39861426, -0.39984535, -0.3981232 ,
           -0.38876243, -0.35298488, -0.35936414, -0.38920991, -0.40590224,
           -0.40736318, -0.40316663, -0.40298403, -0.39821834, -0.39684668,
           -0.37994032, -0.32071932, -0.36007734, -0.35631489, -0.3751666 ,
           -0.39455306, -0.40756206, -0.4116578 , -0.407691  , -0.40524402,
           -0.39893425, -0.31499792, -0.37081046, -0.36776286, -0.38306517,
           -0.4016823 , -0.41746188, -0.41947571, -0.4122563 , -0.40695845,
           -0.39373964, -0.39819061, -0.41112913, -0.41787224, -0.41352376,
           -0.40112084, -0.39164215, -0.39380887, -0.40587672, -0.38928626,
           -0.36933511, -0.36749658, -0.37164245, -0.37851832, -0.39489105,
           -0.3905921 , -0.3793572 , -0.36021624, -0.35846105, -0.36491075,
           -0.37559935, -0.38373461, -0.36757445, -0.36945931, -0.37848095,
           -0.37983738, -0.38301111, -0.36772385, -0.35984961, -0.38521887,
           -0.41257482, -0.41853841, -0.40275782])

    # how about the ambiguous spaxels?
    >>> np.log10(niihamap.value)[masks['ambiguous']['global']]
    array([-0.22853627, -0.22545481, -0.37888335, -0.39616408])


Ambiguous Spaxels
^^^^^^^^^^^^^^^^^

Spaxels that cannot be classified as ``sf``, ``agn``, ``seyfert``, or ``liner`` based on all three BPTs, are classified as ambiguous.  You can determine how ambiguous spaxels were classified in the individual BPT diagrams using the individual BPT masks.

.. code-block:: python

    # get the spaxels classified as ambiguous
    >>> ambig = masks['ambiguous']['global']
    >>> y, x = np.where(ambig)
    >>> print(x, y)
    [11 11 16 17] [13 18 26 26]

    # we have 4 ambiguous spaxels. why are they ambiguous?

    # let's examine the sub-classes in each bpt for these 4 spaxels
    # by filtering the individual BPT boolean maps using the ambiguous spaxel map

    # they are star-forming in the NII BPT
    >>> masks['sf']['nii'][ambig]
    array([False, False,  True,  True], dtype=bool)

    # they are star-forming in the SII BPT
    >>> masks['sf']['sii'][ambig]
    array([ True,  True,  True,  True], dtype=bool)

    # they are not star-forming in the OI BPT
    >>> masks['sf']['oi'][ambig]
    array([False, False, False, False], dtype=bool)

    # they are agn in the OI BPT
    >>> masks['agn']['oi'][ambig]
    array([ True,  True,  True,  True], dtype=bool)

    # If you want a new full 2d-boolean array to use elsewhere, use the bitwise & operator

    >>> niisf_ambig = masks['sf']['nii'] & ambig



Modifying the Plot
^^^^^^^^^^^^^^^^^^

Once you return the BPT figure, you are free to modify it anyway you like. There are different strategies you can try, depending on the complexity of what you want to accomplish. In general, manually modifying the plots requires some knowledge of `matplotlib <https://matplotlib.org/>`_. Let us start by creating a BPT diagram:

.. plot::
    :align: center
    :include-source: True
    :context: reset

    >>> from marvin.tools import Maps
    >>> mm = Maps('8485-1901')
    >>> masks, fig, axes = mm.get_bpt()
    >>> print(fig)
    Figure(850x1000)
    >>> print(axes)
    [<mpl_toolkits.axes_grid1.axes_divider.LocatableAxes object at 0x118bf5d30>,
     <mpl_toolkits.axes_grid1.axes_divider.LocatableAxes object at 0x1192f8a20>,
     <mpl_toolkits.axes_grid1.axes_divider.LocatableAxes object at 0x1193ae6d8>,
     <mpl_toolkits.axes_grid1.axes_divider.LocatableAxes object at 0x119481cc0>]


As we can see, the returned figure is a matplolib `figure <http://https://matplotlib.org/api/figure_api.html?highlight=figure#module-matplotlib.figure>`_ object, while the ``axes`` are a list of ``LocatableAxes``. Matplotlib documentation on ``LocatableAxes`` is scarce, but to most effects they can be considered as normal `axes <https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes>`_ objects.

If you want to modify something in the plot but without changing its main structure, you can use the returned figure. For instance, here we will modify the star forming boundary line in the :math:`\rm [SII]/H\alpha` diagram from solid to dashed, and save the resulting plot as a PNG image

.. plot::
    :align: center
    :include-source: True
    :context: close-figs

    >>> print(fig.axes)
    [<mpl_toolkits.axes_grid1.axes_divider.LocatableAxes at 0x111323d30>,
     <mpl_toolkits.axes_grid1.axes_divider.LocatableAxes at 0x11128b278>,
     <mpl_toolkits.axes_grid1.axes_divider.LocatableAxes at 0x111a18908>,
     <mpl_toolkits.axes_grid1.axes_grid.CbarAxes at 0x111901320>,
     <mpl_toolkits.axes_grid1.axes_grid.CbarAxes at 0x1119b7748>,
     <mpl_toolkits.axes_grid1.axes_grid.CbarAxes at 0x111a52828>,
     <mpl_toolkits.axes_grid1.axes_divider.LocatableAxes at 0x1119de358>,
     <mpl_toolkits.axes_grid1.axes_grid.CbarAxes at 0x111aa0fd0>]
    >>> fig.axes[1].lines[0].set_linestyle('--')
    # fig.savefig('/Users/albireo/Downloads/bpt_new.png')

``fig.axes`` returns a list of four ``LocatableAxes`` (the three BPT diagrams and the 2D representation of the masks) plus a number of ``CbarAxes``. Normally, you can ignore the latter ones. Also, note that if you use the option ``use_oi=False`` when creating the BPT diagram, you will only see three ``LocatableAxes``. We select the  :math:`\rm [SII]/H\alpha` as ``fig.axes[1]``. From there, we can access all the axes attributes and methods. For instance, we can select the first line in the plot ``.lines[0]`` and change its style to dashed ``.set_linestyle('--')``.

Alternatively, you may want to grab one of the axes and modify it, then saving it as a new figure. By itself, matplotlib does not allow to reuse axes in a different figure, so Marvin includes some black magic under the hood to facilitate this

.. plot::
    :align: center
    :include-source: True
    :context: close-figs

    >>> nii_ax = axes[1]
    >>> new_fig = nii_ax.bind_to_figure()

``new_fig`` is now an independent figure that contains the axes for the :math:`\rm [SII]/H\alpha` plot. Let us modify it a bit

.. plot::
    :align: center
    :include-source: True
    :context: close-figs

    >>> ax = new_fig.axes[0]
    >>> ax.set_title('A custom plot')
    >>> for text in ax.texts:
    >>>     text.set_fontsize(20)
    >>> # new_fig.savefig('nii_new.png')

Here we have added a title to the plot, modified the font size of all the texts in the axes, and then saved it as a new image.

.. admonition:: Warning
    :class: warning

    The ``bind_to_figure()`` method is highly experimental. At best it is hacky; at worst unreliable. You should be careful when using it and critically review all plots that you generate. Note that some elements such as legends will be copied, but the styles will not be maintained. All texts and symbols maintain their original sizes, which may not be optimal for the new plot.

Ultimately, you can use the masks to generate brand-new plots with your preferred styles and additional data. The :ref:`BPT module <marvin-utils-bpt>` contains functions to help producing the |kewley2006|_ classification lines. As an example, let us create a simple plot showing the :math:`\rm [NII]/H\alpha` vs :math:`\rm [OIII]/H\beta` classification

.. plot::
    :align: center
    :include-source: True
    :context: reset

    from marvin.tools import Maps
    from matplotlib import pyplot as plt
    from marvin.utils.dap.bpt import kewley_sf_nii, kewley_comp_nii
    import numpy as np

    mm = Maps('8485-1901')

    masks, fig, axes = mm.get_bpt(show_plot=False)

    # Gets the masks for NII/Halpha
    sf = masks['sf']['nii']
    comp = masks['comp']['nii']
    agn = masks['agn']['nii']

    # Gets the necessary maps
    ha = mm['emline_gflux_ha_6564']
    hb = mm['emline_gflux_hb_4862']
    nii = mm['emline_gflux_nii_6585']
    oiii = mm['emline_gflux_oiii_5008']

    # Calculates log(NII/Ha) and log(OIII/Hb)
    log_nii_ha = np.ma.log10(nii.value / ha.value)
    log_oiii_hb = np.ma.log10(oiii.value / hb.value)

    # Creates figure and axes
    fig, ax = plt.subplots()

    # Plots SF, composite, and AGN spaxels using the masks
    ax.scatter(log_nii_ha[sf], log_oiii_hb[sf], c='b')
    ax.scatter(log_nii_ha[comp], log_oiii_hb[comp], c='g')
    ax.scatter(log_nii_ha[agn], log_oiii_hb[agn], c='r')

    # Creates a linspace of points for plotting the classification lines
    xx_sf_nii = np.linspace(-2, 0.045, int(1e4))
    xx_comp_nii = np.linspace(-2, 0.4, int(1e4))

    # Uses kewley_sf_nii and kewley_comp_nii to plot the classification lines
    ax.plot(xx_sf_nii, kewley_sf_nii(xx_sf_nii), 'k-')
    ax.plot(xx_comp_nii, kewley_comp_nii(xx_comp_nii), 'r-')

    ax.set_xlim(-2, 1)
    ax.set_ylim(-1.5, 1.6)

    ax.set_xlabel(r'log([NII]/H$\alpha$)')
    ax.set_ylabel(r'log([OIII]/H$\beta$)')


..    Things to Try
      ^^^^^^^^^^^^^

    Now that you know about Marvin's BPT, try to do these things

    * For a given BPT mask, compute an average spectrum using Marvin Spaxel and the BPT spaxel coordinates.

    Did you do them? :) Now you can contribute your code into Marvin for others to use.  Hurray!


.. |kewley2006| replace:: Kewley et al. (2006)
.. _kewley2006: https://ui.adsabs.harvard.edu/#abs/2006MNRAS.372..961K/abstract

.. |kauffmann2003| replace:: Kauffmann et al. (2003)
.. _kauffmann2003: https://ui.adsabs.harvard.edu/#abs/2003MNRAS.346.1055K/abstract
