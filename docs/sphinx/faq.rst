.. _marvin-faq:

.. important::

    We can use your help to expand this section. If you have encountered an issue
    or have questions that should be addressed here, please
    `submit and issue <https://github.com/sdss/marvin/issues/new>`_.

Frequently Asked Questions and known issues
-------------------------------------------

How do I update marvin?
~~~~~~~~~~~~~~~~~~~~~~~

Just do ``pip install --upgrade sdss-marvin``. Marvin will get updated to the latest
version, along with all the dependencies. If you want to update marvin but keep other
packages in their currrent versions, do
``pip install --upgrade --upgrade-strategy only-if-needed sdss-marvin``. This will only
update dependencies if marvin does need it.

How to test that marvin has been installed correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Marvin is built to have you started with minimum configuration on your part. This means that
marvin is likely to import but maybe not all features will be available. Here are a few commands
you can try that will inform you if there are problems with your installation.

After installing marvin, start a python/ipython session and run::

    import marvin
    print(marvin.config.urlmap)

If you get a dictionary with API routes, marvin is connecting correctly to the API server at
Utah and you can use the remote features. However, if you get ``None``, you may want to
check the steps in :ref:`setup-netrc`.

Matplotlib backend problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some uses have reported that after installing marvin they get an error such as: **Python is not
installed as a framework. The Mac OS X backend will not be able to function correctly if
Python is not installed as a framework.**

This problem is caused by matplotlib not being able to use the MacOS backend if you are using
anaconda. Follow `these instructions
<http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>`_ to fix
the problem. If you do want to use the MacOS backend, consider installing Python using
`homebrew <http://brew.sh/>`_.
