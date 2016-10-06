
def parse_params(request):
    """Parses the mplver and drver from a POST Interaction request."""

    mplver = request.form['mplver'] if 'mplver' in request.form else None
    drver = request.form['drver'] if 'drver' in request.form else None

    return mplver, drver
