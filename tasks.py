# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-10 16:46:40
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-14 19:37:22

from __future__ import absolute_import, division, print_function

import os

from invoke import Collection, task


DIRPATH = '/home/manga/software/git/manga/marvin'
MODULEPATH = '/home/manga/software/git/modulefiles'


@task
def clean_docs(ctx):
    ''' Cleans up the docs '''
    print('Cleaning the docs')
    ctx.run("rm -rf docs/sphinx/_build")


@task
def build_docs(ctx, clean=False):
    ''' Builds the Sphinx docs '''

    if clean:
        print('Cleaning the docs')
        ctx.run("rm -rf docs/sphinx/_build")

    print('Building the docs')
    os.chdir('docs/sphinx')
    ctx.run("make html", pty=True)


@task
def show_docs(ctx):
    """Shows the Sphinx docs"""
    print('Showing the docs')
    os.chdir('docs/sphinx/_build/html')
    ctx.run('open ./index.html')


@task
def clean(ctx):
    ''' Cleans up the crap '''
    print('Cleaning')
    # ctx.run("rm -rf docs/sphinx/_build")
    ctx.run("rm -rf htmlcov")
    ctx.run("rm -rf build")
    ctx.run("rm -rf dist")


@task(clean)
def deploy(ctx):
    ''' Deploy to pypi '''
    print('Deploying to Pypi!')
    ctx.run("python setup.py sdist bdist_wheel --universal")
    # pre-registration is deprecated for new pypi releases [~July 2017]
    # ctx.run("twine register dist/sdss-marvin-*.tar.gz")
    # ctx.run("twine register dist/sdss_marvin-*-none-any.whl")
    ctx.run("twine upload dist/*")


@task
def update_default(ctx, path=None, version=None):
    ''' Updates the default version module file'''

    assert version is not None, 'A version is required to update the default version!'
    assert path is not None, 'A path must be specified!'

    # update default version
    f = open('.version', 'r+')
    data = f.readlines()
    data[1] = 'set ModulesVersion "{0}"\n'.format(version)
    f.seek(0, 0)
    f.writelines(data)
    f.close()


@task
def update_module(ctx, path=None, wrap=None, version=None):
    ''' Update a module file '''

    assert version is not None, 'A version is required to update the module file!'
    assert path is not None, 'A path must be specified!'
    print('Setting up module files!')
    os.chdir(path)
    newfile = 'mangawork.marvin_{0}'.format(version) if wrap else version
    oldfile = 'mangawork.marvin_2.1.3' if wrap else 'master'
    searchline = 'marvin' if wrap else 'version'
    ctx.run('cp {0} {1}'.format(oldfile, newfile))
    f = open('{0}'.format(newfile), 'r+')
    data = f.readlines()
    index, line = [(i, line) for i, line in enumerate(data)
                   if 'set {0}'.format(searchline) in line][0]
    data[index] = 'set {0} {1}\n'.format(searchline, version)
    f.seek(0, 0)
    f.writelines(data)
    f.close()

    # update the default version
    update_default(ctx, path=path, version=newfile)


@task
def update_git(ctx, version=None):
    ''' Update the git package at Utah '''
    assert version is not None, 'A version is required to checkout a new git repo!'
    print('Checking out git tag {0}'.format(version))
    verpath = os.path.join(DIRPATH, version)
    # checkout and setup new git tag
    os.chdir(DIRPATH)
    ctx.run('git clone https://github.com/sdss/marvin.git {0}'.format(version))
    os.chdir(verpath)
    ctx.run('git checkout {0}'.format(version))
    ctx.run('git submodule update --init --recursive')
    # ctx.run('python -c "from get_version import generate_version_py; '
    #         'generate_version_py(\'sdss-marvin\', {0}, False)'.format(version))


@task
def update_current(ctx, version=None):
    ''' Update the current symlink '''
    assert version is not None, 'A version is required to update the current symlink!'
    # reset the current symlink
    os.chdir(DIRPATH)
    ctx.run('rm current')
    ctx.run('ln -s {0} current'.format(version))


@task
def switch_module(ctx, version=None):
    ''' Switch to the marvin module of the specified version and start it '''
    assert version is not None, 'A version is required to setup Marvin at Utah!'
    ctx.run('uwsgi --stop /home/www/sas.sdss.org/mangawork/marvin/pid/uwsgi_marvin.pid')
    ctx.run('module unload wrapmarvin')
    ctx.run('module load wrapmarvin/mangawork.marvin_{0}'.format(version))
    ctx.run('uwsgi /home/manga/software/git/manga/marvin/{0}/python/marvin/web/uwsgi_conf_files/uwsgi_marvin_mangawork.ini'.format(version))


@task
def update_uwsgi(ctx, version=None):
    ''' Reset the uwsgi symlink to the new version and touch the file to Emperor reload Marvin '''
    assert version is not None, 'A version is required to setup Marvin at Utah!'
    os.chdir('/etc/uwsgi/vassals')
    new_path = '/home/manga/software/git/manga/marvin/{0}/python/marvin/web/uwsgi_conf_files/uwsgi_marvin_mangawork.ini'.format(version)
    ctx.run('rm uwsgi_marvin_mangawork.ini')
    ctx.run('ln -s {0} uwsgi_marvin_mangawork.ini'.format(new_path))
    ctx.run('touch uwsgi_marvin_mangawork.ini')


@task
def setup_utah(ctx, version=None):
    ''' Setup the package at Utah and update the release '''
    assert version is not None, 'A version is required to setup Marvin at Utah!'

    # update git
    update_git(ctx, version=version)

    # update_current
    update_current(ctx, version=version)

    # update modules
    marvin = os.path.join(MODULEPATH, 'marvin')
    wrap = os.path.join(MODULEPATH, 'wrapmarvin')
    update_module(ctx, path=marvin, version=version)
    update_module(ctx, path=wrap, wrap=True, version=version)

    # restart the new marvin
    # switch_module(ctx, version=version)
    update_uwsgi(ctx, version=version)
    print('Marvin version {0} is set up!\n'.format(version))
    print('Check for the new Marvin version at the bottom of the Marvin Web main page!')
    # print('Please run ...\n stopmarvin \n module switch wrapmarvin '
    #       'wrapmarvin/mangawork.marvin_{0} \n startmarvin \n'.format(version))


os.chdir(os.path.dirname(__file__))

ns = Collection(clean, deploy, setup_utah)
docs = Collection('docs')
docs.add_task(build_docs, 'build')
docs.add_task(clean_docs, 'clean')
docs.add_task(show_docs, 'show')
ns.add_collection(docs)
updates = Collection('update')
updates.add_task(update_git, 'git')
updates.add_task(update_current, 'current')
updates.add_task(update_module, 'module')
updates.add_task(update_default, 'default')
ns.add_collection(updates)
