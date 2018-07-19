# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-11-21 11:56:56
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-19 15:42:46

from __future__ import print_function, division, absolute_import
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
import traceback


def _indent(text, level=1):
    ''' Format Bintypes '''

    prefix = ' ' * (4 * level)

    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if line.strip() else line)

    return ''.join(prefixed_lines())


def _format_datacubes(datacubes):
    ''' Format Spectra '''

    n_bins = len(datacubes)
    b = datacubes[0]

    yield '.. list-table:: Datacubes'
    yield _indent(':widths: 15 50 50 10 10 20 20')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Description')
    yield _indent('  - Unit')
    yield _indent('  - Ivar')
    yield _indent('  - Mask')
    yield _indent('  - DB')
    yield _indent('  - FITS')

    for datacube in datacubes:
        dbcolumn = '{0}.{1}'.format(datacube.db_table, datacube.db_column())
        yield _indent('* - {0}'.format(datacube.name))
        yield _indent('  - {0}'.format(datacube.description))
        yield _indent('  - {0}'.format(datacube.unit.to_string()))
        yield _indent('  - {0}'.format(datacube.has_ivar()))
        yield _indent('  - {0}'.format(datacube.has_mask()))
        yield _indent('  - {0}'.format(dbcolumn))
        yield _indent('  - {0}'.format(datacube.fits_extension()))
    yield ''


def _format_spectra(spectra):
    ''' Format Spectra '''

    n_bins = len(spectra)
    b = spectra[0]

    yield '.. topic:: Spectra'
    yield '.. list-table:: Spectra'
    yield _indent(':widths: 15 100 20 20 20')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Description')
    yield _indent('  - Unit')
    yield _indent('  - DB')
    yield _indent('  - FITS')

    for spectrum in spectra:
        dbcolumn = '{0}.{1}'.format(spectrum.db_table, spectrum.db_column())
        yield _indent('* - {0}'.format(spectrum.name))
        yield _indent('  - {0}'.format(spectrum.description))
        yield _indent('  - {0}'.format(spectrum.unit.to_string()))
        yield _indent('  - {0}'.format(dbcolumn))
        yield _indent('  - {0}'.format(spectrum.fits_extension()))
    yield ''


def _format_bintypes(bintypes):
    ''' Format Bintypes '''

    n_bins = len(bintypes)
    b = bintypes[0]

    yield '.. list-table:: Bintypes'
    yield _indent(':widths: 15 100 10')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Description')
    yield _indent('  - Binned')
    for bintype in bintypes:
        yield _indent('* - {0}'.format(bintype.name))
        yield _indent('  - {0}'.format(bintype.description))
        yield _indent('  - {0}'.format(bintype.binned))
    yield ''


def _format_templates(templates):
    ''' Format Templates '''

    n_temps = len(templates)
    b = templates[0]

    yield '.. list-table:: Templates'
    yield _indent(':widths: 15 100')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Description')
    for template in templates:
        yield _indent('* - {0}'.format(template.name))
        yield _indent('  - {0}'.format(template.description))
    yield ''


def _format_models(models):
    ''' Format Models '''

    n_temps = len(models)
    b = models[0]

    yield '.. list-table:: Models'
    yield _indent(':widths: 15 100 50 20 15 15')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Description')
    yield _indent('  - Unit')
    yield _indent('  - BinId')
    yield _indent('  - Ivar')
    yield _indent('  - Mask')

    for model in models:
        yield _indent('* - {0}'.format(model.name))
        yield _indent('  - {0}'.format(model.description))
        yield _indent('  - {0}'.format(model.unit))
        yield _indent('  - {0}'.format(model.binid.name))
        yield _indent('  - {0}'.format(model.has_ivar()))
        yield _indent('  - {0}'.format(model.has_mask()))

    yield ''


def _format_properties(properties):
    ''' Format Properties '''

    n_temps = len(properties)
    b = properties[0]

    exts = properties.extensions
    n_exts = len(exts)

    yield '.. list-table:: Properties'
    yield _indent(':widths: 15 100 100 15 15 100 50')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Channels')
    yield _indent('  - Description')
    yield _indent('  - Ivar')
    yield _indent('  - Mask')
    yield _indent('  - DB')
    yield _indent('  - FITS')

    for prop in exts:
        yield _indent('* - {0}'.format(prop.name))
        if 'MultiChannelProperty' in str(prop.__class__):
            channels = ', '.join([c.name for c in prop.channels])
            dbcolumn = ', '.join(['{0}.{1}'.format(prop.db_table, c) for c in prop.db_columns()])
        else:
            channels = prop.channel
            dbcolumn = '{0}.{1}'.format(prop.db_table, prop.db_column())

        yield _indent('  - {0}'.format(channels))
        yield _indent('  - {0}'.format(prop.description))
        yield _indent('  - {0}'.format(prop.ivar))
        yield _indent('  - {0}'.format(prop.mask))
        yield _indent('  - {0}'.format(dbcolumn))
        yield _indent('  - {0}'.format(prop.fits_extension()))

    yield ''


def _format_parameters(parameters):
    ''' Format Query Parameters '''

    yield '.. topic:: Query Parameters'
    yield '.. list-table:: Query Parameters'
    yield _indent(':widths: 25 50 10 20 20 20 20')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Group')
    yield _indent('  - Full Name')
    yield _indent('  - Best')
    yield _indent('  - Name')
    yield _indent('  - DB Schema')
    yield _indent('  - DB Table')
    yield _indent('  - DB Column')

    for param in parameters:
        yield _indent('* - {0}'.format(param.group))
        yield _indent('  - {0}'.format(param.full))
        yield _indent('  - {0}'.format(param.best))
        yield _indent('  - {0}'.format(param.name))
        yield _indent('  - {0}'.format(param.db_schema))
        yield _indent('  - {0}'.format(param.db_table))
        yield _indent('  - {0}'.format(param.db_column))
    yield ''


def _format_schema(schema):
    ''' Format a maskbit schema '''

    schema_dict = schema.to_dict()
    indices = schema_dict['bit'].keys()

    yield '.. list-table:: Schema'
    yield _indent(':widths: 5 50 50')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Bit')
    yield _indent('  - Label')
    yield _indent('  - Description')

    for index in indices:
        yield _indent('* - {0}'.format(schema_dict['bit'][index]))
        yield _indent('  - {0}'.format(schema_dict['label'][index].strip()))
        yield _indent('  - {0}'.format(schema_dict['description'][index].strip()))
    yield ''


def _format_bitmasks(maskbit, bittype):
    ''' Format Maskbits '''

    for name, mask in maskbit.items():
        if bittype.lower() in name.lower():
            #yield '.. program:: {0}'.format(name)
            yield '{0}: {1}'.format(name, mask.description)
            yield ''
            for line in _format_schema(mask.schema):
                yield line


def _format_vacs(vacs, release):
    ''' Format a vac schema '''

    yield '.. list-table:: VACs'
    yield _indent(':widths: 20 10 50')
    yield _indent(':header-rows: 1')
    yield ''
    yield _indent('* - Name')
    yield _indent('  - Version')
    yield _indent('  - Description')

    for vac in vacs:
        yield _indent('* - {0}'.format(vac.name))
        yield _indent('  - {0}'.format(vac.version[release]))
        yield _indent('  - {0}'.format(vac.description))
    yield ''


def _format_command(name, command, **kwargs):
    """Format the output of `click.Command`."""

    # docstring
    # yield command.__doc__
    # yield ''

    # bintypes
    if 'bintypes' in kwargs:
        for line in _format_bintypes(command.bintypes):
            yield line

    # templates
    if 'templates' in kwargs:
        for line in _format_templates(command.templates):
            yield line

    # models
    if 'models' in kwargs:
        for line in _format_models(command.models):
            yield line

    # properties
    if 'properties' in kwargs:
        for line in _format_properties(command.properties):
            yield line

    # spectra
    if 'spectra' in kwargs:
        for line in _format_spectra(command.spectra):
            yield line

    # datacubes
    if 'datacubes' in kwargs:
        for line in _format_datacubes(command.datacubes):
            yield line

    # query parameters
    if 'parameters' in kwargs:
        for line in _format_parameters(command.parameters):
            yield line

    # bitmasks
    if 'bitmasks' in kwargs:
        for line in _format_bitmasks(command.bitmasks, kwargs.get('bittype', None)):
            yield line

    # vacs
    if 'vac' in kwargs:
        vac_release = kwargs.get('vac', None)
        if vac_release and vac_release in command:
            vacdm = command[vac_release]
            for line in _format_vacs(vacdm.vacs, vacdm.release):
                yield line


class DataModelDirective(rst.Directive):

    has_content = False
    required_arguments = 1
    option_spec = {
        'prog': directives.unchanged_required,
        'title': directives.unchanged,
        'bintypes': directives.flag,
        'templates': directives.flag,
        'models': directives.flag,
        'properties': directives.flag,
        'datacubes': directives.flag,
        'spectra': directives.flag,
        'bitmasks': directives.flag,
        'parameters': directives.flag,
        'bittype': directives.unchanged,
        'vac': directives.unchanged,
    }

    def _load_module(self, module_path):
        """Load the module."""

        # __import__ will fail on unicode,
        # so we ensure module path is a string here.
        module_path = str(module_path)

        try:
            module_name, attr_name = module_path.split(':', 1)
        except ValueError:  # noqa
            raise self.error('"{0}" is not of format "module:parser"'.format(module_path))

        try:
            mod = __import__(module_name, globals(), locals(), [attr_name])
        except (Exception, SystemExit) as exc:  # noqa
            err_msg = 'Failed to import "{0}" from "{1}". '.format(attr_name, module_name)
            if isinstance(exc, SystemExit):
                err_msg += 'The module appeared to call sys.exit()'
            else:
                err_msg += 'The following exception was raised:\n{0}'.format(traceback.format_exc())

            raise self.error(err_msg)

        if not hasattr(mod, attr_name):
            raise self.error('Module "{0}" has no attribute "{1}"'.format(module_name, attr_name))

        return getattr(mod, attr_name)

    def _generate_nodes(self, name, command, parent=None, options={}):
        """Generate the relevant Sphinx nodes.
        Format a `click.Group` or `click.Command`.
        :param name: Name of command, as used on the command line
        :param command: Instance of `click.Group` or `click.Command`
        :param parent: Instance of `click.Context`, or None
        :param show_nested: Whether subcommands should be included in output
        :returns: A list of nested docutil nodes
        """

        # Title
        source_name = name

        section = nodes.section(
            '',
            nodes.title(text=name),
            ids=[nodes.make_id(source_name)],
            names=[nodes.fully_normalize_name(source_name)])

        # Summary

        result = statemachine.ViewList()
        lines = _format_command(name, command, **options)
        for line in lines:
            result.append(line, source_name)
        self.state.nested_parse(result, 0, section)

        return [section]

    def run(self):
        self.env = self.state.document.settings.env

        command = self._load_module(self.arguments[0])

        if 'prog' in self.options:
            prog_name = self.options.get('prog')
        else:
            raise self.error(':prog: must be specified')

        return self._generate_nodes(prog_name, command, None, options=self.options)


def setup(app):
    app.add_directive('datamodel', DataModelDirective)



