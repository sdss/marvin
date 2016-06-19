
First Steps
===========

Marvin-tools::
    
    import marvin

Setup config::
    
    from marvin import config
    # Set by DR
    config.setDR('DR13')
    # set by MPL
    config.setMPL('MPL-4')
    # set by DRP and DAP versions
    config.setVersions(drpver='v1_5_1', dapver='1.1.1'

    # set data access mode
    config.mode = 'remote'

Cube::
    
    

Query::
    
    from marvin.tools import Query
    