
.. _marvin-getting_started:

Getting Started
===============

If you have not yet installed Marvin, please follow the :ref:`marvin-installation` instructions before proceeding.  In particular, make sure you have a **netrc** file in your local home directory.  This will enable Marvin to access data remotely, and download files.

Accessing Objects Remotely
--------------------------

From your terminal, type ipython.  Ipython is an Interactive Python shell terminal.  It is recommended to always use ipython instead of python.::

    > ipython

Marvin has a variety of Tools designed to help you access the various MaNGA data products.  Let's access the MaNGA datacube output by the Data Reduction Pipeline (DRP).  The Marvin Cube class is designed to aid your interaction with MaNGA's datacubes.

::

    # import the Marvin Cube tool
    from marvin.tools.cube import Cube

Once the tool is imported, you can instantiate a particular

::

    # instantiate a Marvin Cube for the MaNGA object with plate-ifu 8485-1901
    cube = Cube(plateifu='8485-1901')

    # display a string representation of your cube
    print(cube)
    <Marvin Cube (plateifu='8485-1901', mode='remote', data_origin='api')>

It shows you have created a Marvin Cube for plate-ifu **8485-1901**.  You will also see **mode** and **data_origin** keywords.  These keywords inform you of how your cube was accessed.  You can see that your cube was opened remotely via the Marvin API.  With a cube


Displaying a Plot
-----------------

Downloading Your Object
-----------------------

Accessing Objects Locally
-------------------------

Querying the Sample
-------------------

Download Objects in Bulk
------------------------

Converting to Marvin Objects
----------------------------

Looking at Images
-----------------




