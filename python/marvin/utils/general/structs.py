#!/usr/bin/env python
# encoding: utf-8
#
# structs.py
#
# Created by José Sánchez-Gallego on 17 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from collections import OrderedDict

import six

from fuzzywuzzy import fuzz as fuzz_fuzz
from fuzzywuzzy import process as fuzz_proc


__ALL__ = ['FuzzyDict', 'Dotable', 'DotableCaseInsensitive', 'get_best_fuzzy',
           'FuzzyList']


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
