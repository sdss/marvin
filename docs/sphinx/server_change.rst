
.. _marvin-server-workaround:

Quick Note
----------

As of June 2024, the Marvin Web and API have been migrated to a new server home.  You should see no
change in the use of the Web, however your use of the Marvin python package will need tweaking.

When accessing files remotely, you may encounter the following error message:
::

    BrainError: Requests Http Status Error: 409 Client Error: CONFLICT for url: xxx
    Error: data release None no longer supported by the Marvin API. Update to a later MPL or use Marvin's local file access mode instead.


To resolve this issue, please update to ``sdss-marvin >= 2.8.2``, where the recent server changes are
accounted for.  For package versions ``<= 2.8.1``, use the following workaround:

In your Python session, run the following code::

    from marvin import config
    config.switchSasUrl(sasmode='mirror')


To make this change permanent, create a configuration YAML file at ``~/.marvin/marvin.yml``, add
the following lines inside:
::
    use_mirror: true
    default_release: dr17
