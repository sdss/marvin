#!/usr/bin/env python

from marvin import config
try:
    from sdss_access import Access
except ImportError as e:
    Access = None


def add_data(rsync, release=None, plate=None, ifu=None, exclude=[]):

    drpver, dapver = config.lookUpVersions(release)
    if 'drpall' not in exclude:
        rsync.add('drpall', drpver=drpver)
    if 'dapall' not in exclude:
        rsync.add('dapall', drpver=drpver, dapver=dapver)
    if 'mangacube' not in exclude:
        rsync.add('mangacube', plate=plate, ifu=ifu, drpver=drpver, wave='LOG')
    if 'mangarss' not in exclude:
        rsync.add('mangarss', plate=plate, ifu=ifu, drpver=drpver, wave='LOG')
    if 'mangaimage' not in exclude:
        rsync.add('mangaimage', plate=plate, drpver=drpver, dir3d='stack', ifu=ifu)

    if release == 'MPL-4':
        if 'mangamap' not in exclude:
            rsync.add('mangamap', plate=plate, drpver=drpver, dapver=dapver, ifu=ifu, bintype='*', mode='*', n='**')
        if 'mangadefault' not in exclude:
            rsync.add('mangadefault', plate=plate, drpver=drpver, dapver=dapver, ifu=ifu)
    elif 'mangadap' not in exclude:
        rsync.add('mangadap', plate=plate, drpver=drpver, dapver=dapver, ifu=ifu, daptype='*', mode='MAPS')
        rsync.add('mangadap', plate=plate, drpver=drpver, dapver=dapver, ifu=ifu, daptype='*', mode='LOGCUBE')

    return rsync

# DR17
rsync_access = Access(label='marvin_get_test_data', release='DR17')
rsync_access.remote()
rsync_access = add_data(rsync_access, release='DR17', plate='8485', ifu='1901')
rsync_access = add_data(rsync_access, release='DR17', plate='7443', ifu='12701')
rsync_access.add('mangagema', ver='2.0.2')
rsync_access.add('mangagalaxyzoo', ver='v1_0_1', file='GZD_auto')
rsync_access.add('mangahisum', ver='v2_0_1')
rsync_access.add('mangahispectra', ver='v2_0_1', program='gbt', plateifu='7443-12701')
rsync_access.set_stream()
rsync_access.commit()

# DR15
# rsync_access = Access(label='marvin_get_test_data', release='DR15')
# rsync_access.remote()
# rsync_access = add_data(rsync_access, release='DR15', plate='8485', ifu='1901')
# rsync_access = add_data(rsync_access, release='DR15', plate='7443', ifu='12701')
# rsync_access.set_stream()
# rsync_access.commit()

# MPL-7
# rsync_access = add_data(rsync_access, release='MPL-7', plate='8485', ifu='1901')
# rsync_access = add_data(rsync_access, release='MPL-7', plate='7443', ifu='12701')

# MPL-6
#rsync_access = add_data(rsync_access, release='MPL-6', plate='8485', ifu='1901')
#rsync_access = add_data(rsync_access, release='MPL-6', plate='7443', ifu='12701')

# MPL-5
#rsync_access = add_data(rsync_access, release='MPL-5', plate='8485', ifu='1901')
#rsync_access = add_data(rsync_access, release='MPL-5', plate='7443', ifu='12701')  #, exclude=['mangaimage', 'mangadap'])
#rsync_access = add_data(rsync_access, release='MPL-5', plate='8485', ifu='1901', exclude=['mangacube', 'mangarss', 'mangaimage', 'mangadap'])

# MPL-4
#rsync_access = add_data(rsync_access, release='MPL-4', plate='8485', ifu='1901')
#rsync_access = add_data(rsync_access, release='MPL-4', plate='7443', ifu='12701')  #, exclude=['mangaimage', 'mangamap', 'mangadefault'])
#rsync_access = add_data(rsync_access, release='MPL-4', plate='7443', ifu='12701', exclude=['mangacube', 'mangarss', 'mangaimage', 'mangamap', 'mangadefault'])

# Download
# rsync_access.set_stream()
# rsync_access.commit()
