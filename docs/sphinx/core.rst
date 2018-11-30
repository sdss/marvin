
.. _marvin-core:

Marvin internals (marvin.core)
==============================

This section describes how Marvin works internally. Most users do not need to understand all of these details. However, they are important if you plan to write contributed code or if you are forking Marvin.

Regardless of how you use Marvin, it is always a good idea to have a general understanding of the :ref:`configuration <marvin-config-info>` system and the :ref:`data access modes <marvin-dam>`.

MarvinToolsClass
----------------

`.MarvinToolsClass` is the main class from which most Marvin tool classes subclass (e.g., `~marvin.tools.cube.Cube` or `~marvin.tools.maps.Maps`). It encapsulates the data access :ref:`decision tree <mode-decision-tree>` and implements high level methods such as `.MarvinToolsClass.download` and `.MarvinToolsClass._getFullPath` for using the `tree <https://github.com/sdss/tree>` product to determine paths and :ref:`download <marvin-download-objects>` files. In general, any class that represents a *file* should subclass from `.MarvinToolsClass`.

`.MarvinToolsClass` uses `abc.ABCMeta` as a metaclass to mark `abstractmethods <abc.abstractmethod>`. Any subclass must override and define `.MarvinToolsClass._getFullPath` and `.MarvinToolsClass._set_datamodel`.

Generic methods for `pickling <pickle>` and unpickling the subclassed objects are implemented in `.MarvinToolsClass.save` and `.MarvinToolsClass.restore`. While these work in most cases, depending on the specifics of the subclass some additional handling may be necessary.

.. autoclass:: marvin.tools.core.MarvinToolsClass
   :members: download, save, restore, release, _getFullPath, _set_datamodel, getImage

Mixins
------

Some features are implemented in the form of mixins: parent classes that provide modular functionality. In particular, mixins are provided for accessing the `NSA <http://www.nsatlas.org/data>`_ parameters for an object, and the `DAPall <https://trac.sdss.org/wiki/MANGA/TRM/TRM_ActiveDev/DAPMetaData#DAPall>`_ information.

`.NSAMixIn` must be called with when initialising the child class, and passed the source of the NSA data. `.DAPallMixIn` does not need to be instantiated, and uses `~.MarvinToolsClass.release` to determine the location of the DAPall file.

An example of a class that uses both `.NSAMixIn` and `.DAPallMixIn` would be

.. code-block:: python

   class MyClass(MarvinToolsClass, NSAMixIn, DAPallMixIn):

       def __init__(self, **kwargs):

           MarvinToolsClass.__init__(self, **kwargs)
           NSAMixIn.__init__(self, **kwargs)

       # Overrides abstractmethods
       def _getFullPath(self):
          pass

       def _set_datamodel(self):
          pass

note that we use a direct call to ``__init__`` instead of `super` to make sure that both parent classes are initialised.

.. automodule:: marvin.tools.core
   :members: NSAMixIn, DAPallMixIn


|

Further reading
===============

* The configuration class (:ref:`marvin.config <marvin-config-info>`)
* :ref:`marvin-dam`
* :ref:`marvin-download-objects`
