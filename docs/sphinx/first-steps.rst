
First Steps
===========

Marvin-tools::
    
    import marvin
    
Cube::
        
    from marvin.tools.cube import Cube
    c = Cube(mangaid='')
    c.download()
    c.getSpaxel()
    
Spaxel::
    
    from marvin.tools.spaxel import Spaxel
    sp = Spaxel(mangaid='', x=1, y=2)


Query and Results::
        
    from marvin.tools import doQuery
    q, r = doQuery()
    r.results



Setup config::
    
    from marvin import config
    # Set by DR
    config.setDR('DR13')
    # set by MPL
    config.setMPL('MPL-4')
    # set by DRP and DAP versions
    config.setVersions(drpver='v1_5_1', dapver='1.1.1')

    # set data access mode
    config.mode = 'remote'

