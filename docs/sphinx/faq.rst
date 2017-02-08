.. _marvin-faq:

.. important::

    We can use your help to expand this section. If you have encountered an issue
    or have questions that should be addressed here, please
    `submit and issue <https://github.com/sdss/marvin/issues/new>`_.

Frequently Asked Questions and known issues
-------------------------------------------

How do I update marvin?
^^^^^^^^^^^^^^^^^^^^^^^

Just do ``pip install --upgrade sdss-marvin``. Marvin will get updated to the latest
version, along with all the dependencies. If you want to update marvin but keep other
packages in their currrent versions, do
``pip install --upgrade --upgrade-strategy only-if-needed sdss-marvin``. This will only
update dependencies if marvin does need it.


Permissions Error
^^^^^^^^^^^^^^^^^
If your Marvin installation fails at any point during the pip install process with permissions problems,
try running ``sudo pip install sdss-marvin``.  Note that an Anaconda or Homebrew distribution will not require
permissions when pip installing things, so if you are receiving permissions errors, you may want to check that
you are not using the Mac OSX system version of Python.


How to test that marvin has been installed correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marvin is built to have you started with minimum configuration on your part. This means that
marvin is likely to import but maybe not all features will be available. Here are a few commands
you can try that will inform you if there are problems with your installation.

After installing marvin, start a python/ipython session and run::

    import marvin
    print(marvin.config.urlmap)

If you get a dictionary with API routes, marvin is connecting correctly to the API server at
Utah and you can use the remote features. However, if you get ``None``, you may want to
check the steps in :ref:`setup-netrc`.

Marvin Remote Access Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the above test crashes, or you attempt to use a Marvin Tool remotely, and you see this error::

    AttributeError: 'Extensions' object has no attribute 'get_extension_for_class'

This is an issue with the Urllib and Requests python package.  See `this Issue <https://github.com/sdss/marvin/issues/102>`_ for an
ongoing discussion if this problem has been solved.


Matplotlib backend problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some users have reported that after installing marvin they get an error such as:

**Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if
Python is not installed as a framework.**

This problem is caused by matplotlib not being able to use the MacOS backend if you are using
Anaconda. You need to switch your matplolib backend to ``Agg`` or ``TkAgg``.  Follow `these instructions
<http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>`_ to fix
the problem. If you do want to use the MacOS backend, consider installing Python using
`homebrew <http://brew.sh/>`_.

Web Browser Oddities
^^^^^^^^^^^^^^^^^^^^

If the MPL dropdown list in the top menu bar is blank, or other elements appear to disappear, this is an indication
your browser cache is creating conflicts.  The solution is to clear your browser cache, close and restart your browser from scratch.
You can also clear your browser cookies.

As a reminder, we recommend these browsers for the best Marvin web experience:

* Google Chrome 53+ or higher
* Mozilla Firefox 50+ or higher
* Safari 10+ or Safari Technology Preview



