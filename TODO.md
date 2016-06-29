# TODO / Ideas

## General

- Assign ownership to files / areas of code.
- Use pull requests
- Fix circular imports (either refactor or use import XXX vs from XXX import YYY)
- Add README and CHANGELOG.
- Use TODO in code.
- Add link to issue tracking in Marvin-web/documentation.
- Check valid kwargs for all MarvinToolsClass classes.
- xyorig standards.
- Document attributes
- Maybe geturlmap should not run when loading marvin, but only when the first
    Interaction is called.

- make toggle sasurl function in Config
- make it easier to turn off local db if you have one

## Query

- Make remote mode queries NOT return all the cubes - paginate results
- Setting returntype causes segmentation faults

## Maps

- Attributes for commonly used parameters (basically rework the header into
    something more readable).

## API

- Move api.cube.getSpaxel to api.general and rename it to convertCoords or
    something similar.

## Map

- Make the class MarvinToolsClass.
- Allow to plot in sky coordinates (pywcsgrid2?).
- Allow loading with incomplete but unique category/channel. E.g.,
    category='gflux', channel='ha'.
- Test suite


## DAP zonal queries

- Run benchmarking of unpacking using full MPL4 and 10,000 galaxies.
