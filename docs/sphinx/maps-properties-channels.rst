.. _marvin-maps-properties-channels:

Maps Properties and Channels
============================

.. # code to generate list of properties and channels

    from marvin import config
    from marvin.tools.maps import Maps

    config.setRelease('MPL-4')
    release = ''.join(config.release.lower().split('-'))

    maps = Maps(plateifu='8485-1901')

    print()

    for prop in maps.properties:
        
        channel0 = '' if prop.channels is None else r'``{}``'.format(prop.channels[0])
        if 'emline' in prop.name:
            channel0 = ':ref:`marvin-{}-emline-channels`'.format(release)
        if 'specindex' in prop.name:
            channel0 = ':ref:`marvin-{}-specindex-channels`'.format(release)

        print(r'"``{0}``", "{1}"'.format(prop.name, channel0))

        if (prop.channels is not None) and (len(prop.channels) > 1):
            if ('emline' not in prop.name) and ('specindex' not in prop.name):
                for ch in prop.channels[1:]:
                    print(r'"", "``{}``"'.format(ch))

    print()

    for channel in maps.properties['emline_gflux'].channels:
        print('* ``{}``'.format(channel))

    print()

    for channel in maps.properties['specindex'].channels:
        print('* ``{}``'.format(channel))


MPL-5 Properties and Channels
-----------------------------

.. csv-table::
   :header: "Properties", "Channels"
   :widths: 15, 120

    "``spx_skycoo``", "``on_sky_x``"
    "", "``on_sky_y``"
    "``spx_ellcoo``", "``elliptical_radius``"
    "", "``elliptical_azimuth``"
    "``spx_mflux``", ""
    "``spx_snr``", ""
    "``binid``", ""
    "``bin_lwskycoo``", "``lum_weighted_on_sky_x``"
    "", "``lum_weighted_on_sky_y``"
    "``bin_lwellcoo``", "``lum_weighted_elliptical_radius``"
    "", "``lum_weighted_elliptical_azimuth``"
    "``bin_area``", ""
    "``bin_farea``", ""
    "``bin_mflux``", ""
    "``bin_snr``", ""
    "``stellar_vel``", ""
    "``stellar_sigma``", ""
    "``stellar_sigmacorr``", ""
    "``stellar_cont_fresid``", "``68th_percentile``"
    "", "``99th_percentile``"
    "``stellar_cont_rchi2``", ""
    "``emline_sflux``", ":ref:`marvin-mpl5-emline-channels`"
    "``emline_sew``", ":ref:`marvin-mpl5-emline-channels`"
    "``emline_gflux``", ":ref:`marvin-mpl5-emline-channels`"
    "``emline_gvel``", ":ref:`marvin-mpl5-emline-channels`"
    "``emline_gsigma``", ":ref:`marvin-mpl5-emline-channels`"
    "``emline_instsigma``", ":ref:`marvin-mpl5-emline-channels`"
    "``specindex``", ":ref:`marvin-mpl5-specindex-channels`"
    "``specindex_corr``", ":ref:`marvin-mpl5-specindex-channels`"


.. _marvin-mpl5-emline-channels:

MPL-5 Emline Channels
`````````````````````

* ``oiid_3728``
* ``hb_4862``
* ``oiii_4960``
* ``oiii_5008``
* ``oi_6302``
* ``oi_6365``
* ``nii_6549``
* ``ha_6564``
* ``nii_6585``
* ``sii_6718``
* ``sii_6732``
* ``oii_3727``
* ``oii_3729``
* ``heps_3971``
* ``hdel_4102``
* ``hgam_4341``
* ``heii_4687``
* ``hei_5877``
* ``siii_8831``
* ``siii_9071``
* ``siii_9533``


.. _marvin-mpl5-specindex-channels:

MPL-5 Spectral Index Channels
`````````````````````````````

* ``d4000``
* ``dn4000``



MPL-4 Properties and Channels
-----------------------------

.. csv-table::
   :header: "Properties", "Channels"
   :widths: 15, 120

    "``emline_gflux``", ":ref:`marvin-mpl4-emline-channels`"
    "``emline_gvel``", ":ref:`marvin-mpl4-emline-channels`"
    "``emline_gsigma``", ":ref:`marvin-mpl4-emline-channels`"
    "``emline_instsigma``", ":ref:`marvin-mpl4-emline-channels`"
    "``emline_ew``", ":ref:`marvin-mpl4-emline-channels`"
    "``emline_sflux``", ":ref:`marvin-mpl4-emline-channels`"
    "``stellar_vel``", ""
    "``stellar_sigma``", ""
    "``specindex``", ":ref:`marvin-mpl4-specindex-channels`"
    "``binid``", ""


.. _marvin-mpl4-emline-channels:

MPL-4 Emline Channels
`````````````````````

* ``oiid_3728``
* ``hb_4862``
* ``oiii_4960``
* ``oiii_5008``
* ``oi_6302``
* ``oi_6365``
* ``nii_6549``
* ``ha_6564``
* ``nii_6585``
* ``sii_6718``
* ``sii_6732``


.. _marvin-mpl4-specindex-channels:

MPL-4 Spectral Index Channels
`````````````````````````````

* ``d4000``
* ``caii0p39``
* ``hdeltaa``
* ``cn1``
* ``cn2``
* ``ca4227``
* ``hgammaa``
* ``fe4668``
* ``hb``
* ``mgb``
* ``fe5270``
* ``fe5335``
* ``fe5406``
* ``nad``
* ``tio1``
* ``tio2``
* ``nai0p82``
* ``caii0p86a``
* ``caii0p86b``
* ``caii0p86c``
* ``mgi0p88``
* ``tio0p89``
* ``feh0p99``


How to List Properties and Channels Using Marvin Tools
------------------------------------------------------

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(mangaid='1-209232')
    
    # list properties
    maps.properties
    
    # [<Property name=spx_skycoo, ivar=False, mask=False, n_channels=2>,
    #  <Property name=spx_ellcoo, ivar=False, mask=False, n_channels=2>,
    #  <Property name=spx_mflux, ivar=True, mask=False, n_channels=None>,
    #  <Property name=spx_snr, ivar=False, mask=False, n_channels=None>,
    #  <Property name=binid, ivar=False, mask=False, n_channels=None>,
    #  <Property name=bin_lwskycoo, ivar=False, mask=False, n_channels=2>,
    #  <Property name=bin_lwellcoo, ivar=False, mask=False, n_channels=2>,
    #  <Property name=bin_area, ivar=False, mask=False, n_channels=None>,
    #  <Property name=bin_farea, ivar=False, mask=False, n_channels=None>,
    #  <Property name=bin_mflux, ivar=True, mask=True, n_channels=None>,
    #  <Property name=bin_snr, ivar=False, mask=False, n_channels=None>,
    #  <Property name=stellar_vel, ivar=True, mask=True, n_channels=None>,
    #  <Property name=stellar_sigma, ivar=True, mask=True, n_channels=None>,
    #  <Property name=stellar_sigmacorr, ivar=False, mask=False, n_channels=None>,
    #  <Property name=stellar_cont_fresid, ivar=False, mask=False, n_channels=2>,
    #  <Property name=stellar_cont_rchi2, ivar=False, mask=False, n_channels=None>,
    #  <Property name=emline_sflux, ivar=True, mask=True, n_channels=21>,
    #  <Property name=emline_sew, ivar=True, mask=True, n_channels=21>,
    #  <Property name=emline_gflux, ivar=True, mask=True, n_channels=21>,
    #  <Property name=emline_gvel, ivar=True, mask=True, n_channels=21>,
    #  <Property name=emline_gsigma, ivar=True, mask=True, n_channels=21>,
    #  <Property name=emline_instsigma, ivar=False, mask=False, n_channels=21>,
    #  <Property name=specindex, ivar=True, mask=True, n_channels=2>,
    #  <Property name=specindex_corr, ivar=False, mask=False, n_channels=2>]
    
    # list channels for a property
    maps.properties['emline_gflux'].channels
    
    # ['oiid_3728',
    #  'hb_4862',
    #  'oiii_4960',
    #  'oiii_5008',
    #  'oi_6302',
    #  'oi_6365',
    #  'nii_6549',
    #  'ha_6564',
    #  'nii_6585',
    #  'sii_6718',
    #  'sii_6732',
    #  'oii_3727',
    #  'oii_3729',
    #  'heps_3971',
    #  'hdel_4102',
    #  'hgam_4341',
    #  'heii_4687',
    #  'hei_5877',
    #  'siii_8831',
    #  'siii_9071',
    #  'siii_9533']


|