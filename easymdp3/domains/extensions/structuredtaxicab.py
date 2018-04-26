from collections import namedtuple
from itertools import product

from easymdp3.domains.taxicab import TaxiCabMDP
from easymdp3.domains.taxicab.utils import \
    get_interior_tiles, get_building_walls

class StructuredTaxiCab(TaxiCabMDP):
    def __init__(self,
                 width=6, height=6,
                 walls=None,
                 building_params: "walls defined by corners "
                                  "with exit/entrances" = None,
                 locations=None,
                 init_location=(0, 3),
                 init_passengers=None,
                 step_cost=-1
                 ):
        if walls is None:
            walls = []

        tiles_to_regions = {}
        regions_to_tiles = {}
        outside_tiles = set(product(range(width), range(height)))
        if building_params is not None:
            for bi, bparams in enumerate(building_params):
                bwalls = get_building_walls(**bparams)
                interior_tiles = set(get_interior_tiles(bparams['corners']))
                regions_to_tiles['building_%d' % bi] = interior_tiles
                outside_tiles.difference_update(interior_tiles)
                for t in interior_tiles:
                    tiles_to_regions[t] = 'building_%d' % bi
                walls.extend(bwalls)
        else:
            building_params = []

        regions_to_tiles['outside'] = outside_tiles
        for t in outside_tiles:
            tiles_to_regions[t] = 'outside'

        regions_to_locations = {}
        for loc in locations:
            reg = tiles_to_regions[loc]
            regions_to_locations[reg] = regions_to_locations.get(reg, set([]))
            regions_to_locations[reg].add(loc)

        self.building_params = building_params
        self.tiles_to_regions = tiles_to_regions
        self.regions_to_tiles = regions_to_tiles
        self.regions_to_locations = regions_to_locations

        TaxiCabMDP.__init__(self, width=width, height=height, walls=walls,
                            locations=locations, init_location=init_location,
                            init_passengers=init_passengers,
                            step_cost=step_cost)