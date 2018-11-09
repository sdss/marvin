#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-08-17
# @Filename: structs.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-08 18:58:09


from __future__ import absolute_import, division, print_function

import gzip
import tempfile
from collections import OrderedDict
from contextlib import contextmanager

import six
from fuzzywuzzy import fuzz as fuzz_fuzz
from fuzzywuzzy import process as fuzz_proc


__ALL__ = ['FuzzyDict', 'Dotable', 'DotableCaseInsensitive', 'get_best_fuzzy',
           'FuzzyList', 'string_folding_wrapper', 'gunzip']


class Dotable(dict):
    """A custom dict class that allows dot access to nested dictionaries.

    Copied from http://hayd.github.io/2013/dotable-dictionaries/. Note that
    this allows you to use dots to get dictionary values, but not to set them.

    """

    def __getattr__(self, value):
        if '__' in value:
            return dict.__getattr__(self, value)
        else:
            return self.__getitem__(value)

    # def __init__(self, d):
    #     dict.__init__(self, ((k, self.parse(v)) for k, v in d.iteritems()))

    @classmethod
    def parse(cls, v):
        if isinstance(v, dict):
            return cls(v)
        elif isinstance(v, list):
            return [cls.parse(i) for i in v]
        else:
            return v


class DotableCaseInsensitive(Dotable):
    """Like dotable but access to attributes and keys is case insensitive."""

    def _match(self, list_of_keys, value):

        lower_values = [str(xx).lower() for xx in list_of_keys]
        if value.lower() in lower_values:
            return list_of_keys[lower_values.index(value.lower())]
        else:
            return False

    def __getattr__(self, value):
        if '__' in value:
            return super(DotableCaseInsensitive, self).__getattr__(value)
        return self.__getitem__(value)

    def __getitem__(self, value):
        key = self._match(list(self.keys()), value)
        if key is False:
            raise KeyError('{0} key or attribute not found'.format(value))
        return dict.__getitem__(self, key)


def get_best_fuzzy(value, choices, min_score=75, scorer=fuzz_fuzz.WRatio, return_score=False):
    """Returns the best match in a list of choices using fuzzywuzzy."""

    if not isinstance(value, six.string_types):
        raise ValueError('invalid value. Must be a string.')

    if len(value) < 3:
        raise ValueError('your fuzzy search value must be at least three characters long.')

    if len(choices) == 0:
        raise ValueError('choices cannot be an empty list.')

    # If the value contains _ivar or _mask this is probably and incorrect use
    # of the fuzzy feature. We raise an error.
    if '_ivar' in value:
        raise ValueError('_ivar not allowd in search value.')
    elif '_mask' in value:
        raise ValueError('_mask not allowd in search value.')

    bests = fuzz_proc.extractBests(value, choices, scorer=scorer, score_cutoff=min_score)

    if len(bests) == 0:
        best = None
    elif len(bests) == 1:
        best = bests[0]
    else:
        if bests[0][1] == bests[1][1]:
            best = None
        else:
            best = bests[0]

    if best is None:
        raise ValueError('cannot find a good match for {0!r}. '
                         'Your input value is too ambiguous.'.format(value))

    return best if return_score else best[0]


class FuzzyDict(OrderedDict):
    """A dotable dictionary that uses fuzzywuzzy to select the key."""

    def __getattr__(self, value):
        if '__' in value:
            return super(FuzzyDict, self).__getattr__(value)
        return self.__getitem__(value)

    def __getitem__(self, value):

        if not isinstance(value, six.string_types):
            return self.values()[value]

        if value in self.keys():
            return dict.__getitem__(self, value)

        best = get_best_fuzzy(value, self.keys())

        return dict.__getitem__(self, best)

    def __dir__(self):

        return list(self.keys())


class FuzzyList(list):
    """A list that uses fuzzywuzzy to select the item.

    Parameters:
        the_list (list):
            The list on which we will do fuzzy searching.
        use_fuzzy (function):
            A function that will be used to perform the fuzzy selection
    """

    def __init__(self, the_list, use_fuzzy=None):

        self.use_fuzzy = use_fuzzy if use_fuzzy else get_best_fuzzy

        list.__init__(self, the_list)

    def mapper(self, item):
        """The function that maps each item to the querable string."""

        return str(item)

    def __eq__(self, value):

        self_values = [self.mapper(item) for item in self]

        try:
            best = self.use_fuzzy(value, self_values)
        except ValueError:
            # Second pass, using underscores.
            best = self.use_fuzzy(value.replace(' ', '_'), self_values)

        return self[self_values.index(best)]

    def __contains__(self, value):

        if not isinstance(value, six.string_types):
            return super(FuzzyList, self).__contains__(value)

        try:
            self.__eq__(value)
            return True
        except ValueError:
            return False

    def __getitem__(self, value):

        if isinstance(value, six.string_types):
            return self == value
        else:
            return list.__getitem__(self, value)

    def __getattr__(self, value):

        self_values = [super(FuzzyList, self).__getattribute__('mapper')(item)
                       for item in self]

        if value in self_values:
            return self[value]

        return super(FuzzyList, self).__getattribute__(value)

    def __dir__(self):

        return [self.mapper(item) for item in self]


class OrderedDefaultDict(FuzzyDict):

    def __init__(self, default_factory=None, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        result = self[key] = self.default_factory()
        return result


class StringFolder(object):
    """
    Class that will fold strings. See 'fold_string'.
    This object may be safely deleted or go out of scope when
    strings have been folded.

    Credit: Ben Last
    http://dev.mobify.com/blog/sqlalchemy-memory-magic/

    """
    def __init__(self):
        self.unicode_map = {}

    def fold_string(self, s):
        """
        Given a string (or unicode) parameter s, return a string object
        that has the same value as s (and may be s). For all objects
        with a given value, the same object will be returned. For unicode
        objects that can be coerced to a string with the same value, a
        string object will be returned.
        If s is not a string or unicode object, it is returned unchanged.
        :param s: a string or unicode object.
        :return: a string or unicode object.
        """
        # If s is not a string or unicode object, return it unchanged
        if not isinstance(s, six.string_types):
            return s

        # If s is already a string, then str() has no effect.
        # If s is Unicode, try and encode as a string and use intern.
        # If s is Unicode and can't be encoded as a string, this try
        # will raise a UnicodeEncodeError.
        try:
            return six.moves.intern(str(s))
        except UnicodeEncodeError:
            # Fall through and handle s as Unicode
            pass

        # Look up the unicode value in the map and return
        # the object from the map. If there is no matching entry,
        # store this unicode object in the map and return it.
        t = self.unicode_map.get(s, None)
        if t is None:
            # Put s in the map
            t = self.unicode_map[s] = s
        return t


def string_folding_wrapper(results, keys=None):
    """
    This generator yields rows from the results as tuples,
    with all string values folded.

    Credit: Ben Last
    http://dev.mobify.com/blog/sqlalchemy-memory-magic/
    """
    # Get the list of keys so that we build tuples with all
    # the values in key order.

    if keys is None:
        try:
            keys = results.keys()
        except AttributeError as e:
            print('No keys are accessible.  Cannot fold strings!: {0}'.format(e))
            yield results

    folder = StringFolder()
    for row in results:
        yield tuple(
            folder.fold_string(row.__getattribute__(key))
            for key in keys
        )


@contextmanager
def gunzip(filename):

    temp_file = tempfile.NamedTemporaryFile(mode='wb')
    temp_file.file.write(gzip.GzipFile(filename).read())

    try:
        yield temp_file
    finally:
        temp_file.close()
