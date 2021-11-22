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
import six

import marvin
import marvin.tools.plate
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier
from astropy.io import fits
from functools import wraps

import sdss_access.path
import sdss_access.sync


__ALL__ = ['VACContainer', 'VACMixIn']


def check_for_vac(f):
    ''' Decorator to check for and download VAC '''
    @wraps(f)
    def decorated_function(inst, *args, **kwargs):
        if 'path' in kwargs and kwargs['path']:
            for kw in kwargs['path'].split('/'):
                if len(kw) == 0:
                    continue
                var, value = kw.split('=')
                kwargs[var] = value
        kwargs.pop('path')
        return f(inst, *args, **kwargs)
    return decorated_function


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
    # Set this is True on your VAC to exclude it from Marvin
    _hidden = False

    # The name and description of the VAC.
    name = None
    description = None
    
    # custom data container for VAC data in summary file(s)
    # used by tools.vacs.VACs
    data_container = None

    def __init__(self):

        if not sdss_access.sync.Access:
            raise MarvinError('sdss_access is not installed')
        else:
            self._release = marvin.config.release
            # is_public = 'DR' in self._release
            # rsync_release = self._release.lower() if is_public else None
            self.rsync_access = sdss_access.sync.Access(release=self._release)

        # file path for VAC summary file
        self.summary_file = None
        self.set_summary_file(marvin.config.release)

    def __repr__(self):
        return '<VAC (name={0}, description={1})>'.format(self.name, self.description)

    @abc.abstractmethod
    def get_target(self, parent_object):
        """Returns VAC data that matches the `parent_object` target.

        This method must be overridden in each subclass of `VACMixIn`. Details
        will depend on the exact implementation and the type of VAC, but in
        general each version of this method must:

        * Check whether the VAC file exists locally.
        * If it does not, download it using `~VACMixIn.download_vac`.
        * Open the file using the appropriate library.
        * Retrieve the VAC data matching ``parent_object``. Usually one will
          use attributes in ``parent_object`` such as ``.mangaid`` or
          ``.plateifu`` to perform the match.
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
            to `~VACMixIn.get_target` when the subclass of `VACMixIn` is
            called.

        Returns
        -------
        vac_container : object
            An instance of a class that contains just a list of properties, one
            for to each on of the VACs that subclass from `VACMixIn`.

        """

        vac_container = VACContainer()

        for subvac in VACMixIn.__subclasses__():
            # check if VAC is hidden
            if subvac._hidden:
                continue

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
                        property(lambda self, sv=subvac: sv().get_target(parent_object)))

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

        # return the full local path to the file
        path = self.rsync_access.full(name, **path_params)
        return path
        # # check for and expand any wildcards present in the path_params
        # if self.rsync_access.any(name, **path_params):
        #     files = self.rsync_access.expand(name, **path_params)
        #     return files[0]
        # else:
        #     return False

    def file_exists(self, path=None, name=None, path_params={}):
        """Check whether a file exists locally"""

        # use the filepath if present
        if path:
            return os.path.exists(path)

        # otherwise use name and path_params
        if name is None:
            name = self.name

        if os.path.exists(self.get_path(name=name, path_params=path_params)):
            return True

        return False

    def check_vac(self, summary_file):
        ''' Checks the summary file for existence '''
        pass

    @abc.abstractmethod
    def set_summary_file(self, release):
        """ Sets the VAC summary file

        This method must be overridden in each subclass of `VACMixIn`. Details
        will depend on the exact implementation and the type of VAC, but in
        general each version of this method must:

        * Access the version of your VAC matching the current ``release``
        * Define a dictionary of keyword parameters that defines the `tree` path
        * Use `~VACMixIn.get_path` to construct the VAC path
        * Set that path to the `~VACMixIn.summary_file` attribute

        Setting a VAC summary file allows the `~marvin.tools.vacs.VACs` tool to load
        the full VAC data.  If the VAC does not contain a summary file, this method
        should `pass` or return `None`.

        """
        pass

    def update_path_params(self, params):
        ''' Update the path_params dictionary with additional parameters '''

        assert isinstance(params, dict), 'input parameters must be a dictionary'
        self.path_params.update(params)

    def get_ancillary_file(self, name, path_params={}):
        ''' Get a path to an ancillary VAC file '''

        path = self.get_path(name, path_params=path_params)
        if not path:
            path = self.download_vac(name, path_params=path_params)
        return path


class VACTarget(object):
    ''' Customization Class to allow for returning complex target data

    This parent class provides a framework for returning more complex data associated
    with a given target observation, for example ancillary spectral or image data.  In these
    cases, returning a target row from the main VAC summary file, or a simple dictionary of values
    may not be sufficient.  This class can be subclassed and customized to return any
    extra functionality or data.

    When used, this class provides convenient access to the underlying VAC data as well
    as a boolean to indicate if the given target is included in the VAC.

    Parameters:
        targetid (str):
            The target id, usually plateifu or mangaid.  Required.
        vacfile (str):
            The path to the VAC summary file.  Required.

    Attributes:
        targetid (str):
            The plateifu or mangaid target designation
        data (row):
            The extracted row VAC data for the provided targetid
        _data (HDU):
            the first data HDU of the summary VAC FITS file
        _indata (bool):
            A boolean indicating if the target is included in the VAC

    To use, subclass this class, add a new `__init__` method.  Make sure to call the original
    class's `__init__` method with `super`.

    .. code-block:: python

        from marvin.contrib.vacs.base import VACTarget

        class ExampleTarget(VACTarget):

            def __init__(self, targetid, vacfile):
                super(ExampleTarget, self).__init__(targetid, vacfile)

    Further customization can now be done, e.g. adding new parameters in the initializtion of the
    object, adding new methods or attributes, or overriding existing methods, e.g. to customize the
    return `data` attribute.

    To access a single HDU from the VAC, use the `_get_data()` method.  If you need to access the entire file,
    use the `_open_file()` method.

    '''

    def __init__(self, targetid, vacfile, **kwargs):
        self.targetid = targetid
        self._ttype = parseIdentifier(targetid)
        assert self._ttype in ['plateifu', 'mangaid'], 'Input targetid must be a valid plateifu or mangaid'
        self._vacfile = vacfile
        self._data = self._get_data(self._vacfile)
        self._indata = targetid in self._data[self._ttype]

    def __repr__(self):
        return 'Target({0})'.format(self.targetid)

    @property
    def data(self):
        ''' The data row from a VAC for a specific targetid '''
        if not self._indata:
            return "No data exists for {0}".format(self.targetid)

        idx = self._data[self._ttype] == self.targetid
        return self._data[idx]

    @staticmethod
    def _open_file(vacfile):
        ''' Opens the full FITS VAC file '''
        return fits.open(vacfile)

    def _get_data(self, vacfile=None, ext=1):
        ''' Get only the data from the VAC file from a given extension '''
        if not vacfile:
            vacfile = self._vacfile
        return fits.getdata(vacfile, ext)

