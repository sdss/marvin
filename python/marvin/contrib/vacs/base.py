# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 17:01:09
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last Modified time: 2018-10-17 00:22:19

from __future__ import absolute_import, division, print_function

import abc
import os
import time

import sdss_access.path
import sdss_access.sync
import six

import marvin
import marvin.tools.plate
from marvin.core.exceptions import MarvinError


__ALL__ = ['VACContainer', 'VACMixIn']


class VACContainer(object):

    def __repr__(self):
        return '<VACContainer ({0})>'.format(', '.join(map(repr, list(self))))

    def __dir__(self):
        props = []
        for value in self.__class__.__dict__.keys():
            if not value.startswith('_'):
                props.append(value)
        return props

    def __getitem__(self, value):
        return getattr(self, value)

    def __iter__(self):
        for value in self.__dir__():
            yield value


class VACMixIn(object, six.with_metaclass(abc.ABCMeta)):
    """MixIn  that allows VAC integration in Marvin.

    This parent class provides common tools for downloading data using
    sdss_access or directly from the sandbox. `~VACMixIn.get_vacs` returns a
    container with properties pointing to all the VACs that subclass from
    `.VACMixIn`. In general, VACs can be added to a class in the following way:

    .. code-block:: python

        from marvin.contrib.vacs.base import VACMixIn

        class Maps(MarvinToolsClass):

            def __init__(self, *args, **kwargs):

                ...

                self.vacs = VACMixIn.get_vacs(self)

    and then the VACs can be accessed as properties in ``my_map.vacs``.

    """

    # The name and description of the VAC.
    name = None
    description = None

    def __init__(self):

        if not sdss_access.sync.RsyncAccess:
            raise MarvinError('sdss_access is not installed')
        else:
            self._release = marvin.config.release
            is_public = 'DR' in self._release
            rsync_release = self._release.lower() if is_public else None
            self.rsync_access = sdss_access.sync.RsyncAccess(public=is_public, release=rsync_release)

    def __repr__(self):
        return '<VAC (name={0}, description={1})>'.format(self.name, self.description)

    @abc.abstractmethod
    def get_data(self, parent_object):
        """Returns VAC data that matches the `parent_object` target.

        This method must be overridden in each subclass of `VACMixIn`. Details
        will depend on the exact implementation and the type of VAC, but in
        general each version of this method must:

        * Check whether the VAC file exists locally.
        * If it does not, download it using `~VACMixIn.download_vac`.
        * Open the file using the appropriate library.
        * Retrieve the VAC data matching ``parent_object``. Usually one will
          use attributes in ``parent_object`` such as ``.mangaid`` or
          ``.plate`` to perform the match.
        * Return the VAC data in whatever format is appropriate.

        """

        pass

    @staticmethod
    def get_vacs(parent_object):
        """Returns a container with all the VACs subclassing from `VACMixIn`.

        Because this method loops over ``VACMixIn.__subclasses__()``, all the
        class that inherit from `VACMixIn` and that must be included in the
        container need to have been imported before calling
        `~VACMixIn.get_vacs`.

        Parameters
        ----------
        parent_object : object
            The object to which the VACs are being attached. It will be passed
            to `~VACMixIn.get_data` when the subclass of `VACMixIn` is
            called.

        Returns
        -------
        vac_container : object
            An instance of a class that contains just a list of properties, one
            for to each on of the VACs that subclass from `VACMixIn`.

        """

        vac_container = VACContainer()

        for subvac in VACMixIn.__subclasses__():

            # Excludes VACs from showing up in Plate
            if issubclass(parent_object.__class__, marvin.tools.plate.Plate):
                continue

            # Only shows VACs if in the include list.
            if (hasattr(subvac, 'include') and subvac.include is not None and
                    not issubclass(parent_object.__class__, subvac.include)):
                continue

            # We need to set sv=subvac in the lambda function to prevent
            # a cell-var-from-loop issue.
            if parent_object._release in subvac.version:
                setattr(VACContainer, subvac.name,
                        property(lambda self, sv=subvac: sv().get_data(parent_object)))

        return vac_container

    def download_vac(self, name=None, path_params={}, verbose=True):
        """Download the VAC using rsync and returns the local path."""

        if name is None:
            name = self.name

        assert name in self.rsync_access.templates, 'VAC path has not been set in the tree.'

        if verbose:
            marvin.log.info('downloading file for VAC {0!r}'.format(self.name))

        self.rsync_access.remote()
        self.rsync_access.add(name, **path_params)
        self.rsync_access.set_stream()
        self.rsync_access.commit()
        paths = self.rsync_access.get_paths()

        # adding a millisecond pause for download to finish and file existence to register
        time.sleep(0.001)

        return paths[0]  # doing this for single files, may need to change

    def get_path(self, name=None, path_params={}):
        """Returns the local VAC path or False if it does not exist."""

        if name is None:
            name = self.name

        # check for and expand any wildcards present in the path_params
        if self.rsync_access.any(name, **path_params):
            files = self.rsync_access.expand(name, **path_params)
            return files[0]
        else:
            return False

    def file_exists(self, name=None, path_params={}):
        """Check whether a file exists."""

        if name is None:
            name = self.name

        if os.path.exists(self.get_path(name=name, path_params=path_params)):
            return True

        return False
