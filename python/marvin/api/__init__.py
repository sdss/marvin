
def parse_params(request):
    """Parses the drpver and dapver from a POST Interaction request."""

    drpver = request.form['drpver'] if 'drpver' in request.form else None
    dapver = request.form['dapver'] if 'dapver' in request.form else None

    return drpver, dapver
