import re

specindex_types = ['D4000', 'CaII0p39', 'HDeltaA', 'CN1', 'CN2', 'Ca4227', 'HGammaA', 'Fe4668',
                   'Hb', 'Mgb', 'Fe5270', 'Fe5335', 'Fe5406', 'NaD', 'TiO1', 'TiO2', 'NaI0p82',
                   'CaII0p86A', 'CaII0p86B', 'CaII0p86C', 'MgI0p88', 'TiO0p89', 'FeH0p99']

stkin_param_aliases = dict(VEL=['VEL', 'VELOCITY'],
                           SIGMA=['SIGMA', 'SIG'])

emline_name_aliases = dict(OIId3728=['OIId3728', 'OIId-3728', 'OII-3728', 'OII', 'OIId'],
                           Hb4862=['Hb4862', 'Hb-4862', 'Hb', 'Hbeta'],
                           OIII4960=['OIII4960', 'OIII-4960'],
                           OIII5008=['OIII5008', 'OIII-5008'],
                           NII6549=['NII6549', 'NII-6549'],
                           Ha6564=['Ha6564', 'Ha-6564', 'Ha', 'Halpha'],
                           NII6585=['NII6585', 'NII-6585'],
                           SII6718=['SII6718', 'SII-6718'],
                           SII6732=['SII6732', 'SII-6732'])
emline_param_aliases = dict(GFLUX=['GFLUX', 'FLUX'],
                            GVEL=['GVEL', 'VEL', 'VELOCITY'],
                            GSIMGA=['GSIGMA', 'SIG', 'SIGMA'],
                            EW=['EW', 'SEW'],
                            SFLUX=['SFLUX'],
                            INSTSIGMA=['INSTSIGMA'])

sp_lower = [it.lower() for it in specindex_types]
stkin_flat_lower = [i.lower() for v in stkin_param_aliases.values() for i in v]
emline_flat_lower = [i.lower() for v in emline_name_aliases.values() for i in v]
emline_param_flat_lower = [i.lower() for v in emline_param_aliases.values() for i in v]


def assign_category(name):
    category = None
    for frag in name.split('_'):
        nm_low = frag.lower()
        if nm_low in sp_lower:
            category = 'specindex'
            # Hb can be specindex or emline, so try finding an emline parameter in name.
            if nm_low == 'hb':
                for frag2 in name.split('_'):
                    if frag2.lower() in emline_param_flat_lower:
                        category = 'emline'
        elif nm_low == 'st':
            category = 'stellar_kin'
        elif nm_low in emline_flat_lower:
            category = 'emline'
    if category is None:
        raise TypeError
    return category


def assign_type(name, category):
    ctype = None
    for frag in name.split('_'):
        nm_low = frag.lower()
        if category == 'specindex':
            for sp, sp_low in zip(specindex_types, sp_lower):
                if nm_low == sp_low:
                    ctype = sp
        elif category == 'emline':
            for kk, vv in emline_name_aliases.items():
                for it in vv:
                    if nm_low == it.lower():
                        # split name--rest_wavelength string
                        ctype = list(re.findall(r'(\w+?)(\d+)', kk)[0])
                        ctype[1] = int(ctype[1])
    return ctype


def assign_parameter(name, category):
    param = None
    for frag in name.split('_'):
        nm_low = frag.lower()
        if category == 'stellar_kin':
            for kk, vv in stkin_param_aliases.items():
                for it in vv:
                    for frag2 in name.split('_'):
                        if frag2.lower() == it.lower():
                            param = kk
        elif category == 'emline':
            for kk, vv in emline_param_aliases.items():
                for it in vv:
                    if nm_low == it.lower():
                        param = kk
    return param


def assign_value_type(name):
    if 'ivar' in name:
        value_type = 'ivar'
    elif 'mask' in name:
        value_type = 'mask'
    else:
        value_type = 'value'
    return value_type


def join_type_conditions(ctype, category):
    if isinstance(ctype, str):
        ctype = [ctype]

    if ctype is None:
        conditions = None
    else:
        conditions = []
        for ct in ctype:
            if category == 'specindex':
                column = 'specindex_type.name'
            elif category == 'emline':
                if isinstance(ct, str):
                    column = 'emline_type.name'
                elif isinstance(ct, int):
                    column = 'emline_type.rest_wavelength'
            try:
                quote = "'" if isinstance(ct, str) else ''
                ctype_str = quote + str(ct) + quote
                conditions += [' '.join((column, '==', ctype_str))]
            except TypeError:
                conditions = None
    return conditions


def join_parameter_conditions(param, category):
    if category == 'stellar_kin':
        column = 'stellar_kin_parameter.name'
    elif category == 'emline':
        column = 'emline_parameter.name'
    try:
        param_str = "'" + param + "'"
        conditions = [' '.join((column, '==', param_str))]
    except TypeError:
        conditions = None
    return conditions


def expand(name, operator, value):
    category = assign_category(name)
    ctype = assign_type(name, category)
    param = assign_parameter(name, category)
    value_type = assign_value_type(name)
    value_table = '.'.join((category, value_type))
    # create searchfilter
    conditions = []
    type_condition = join_type_conditions(ctype, category)
    param_condition = join_parameter_conditions(param, category)
    value_condition = ' '.join((value_table, operator, value))
    for cn in (type_condition, param_condition, value_condition):
        if cn is not None:
            if isinstance(cn, list):
                for it in cn:
                    conditions.append(it)
            else:
                conditions.append(cn)
    searchfilter = ' and '.join(conditions)
    return searchfilter
