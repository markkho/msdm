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

# class StructuredTaxiCab(TaxiCabMDP):
#     # ================================================ #
#     #    Structured Taxi-Cab States and State Tuples   #
#     # ================================================ #
#     RegionTuple = namedtuple(
#         'Region', ['tiles', 'subregions', 'superregion', 'i']
#     )
#
#     StateTuple = namedtuple(
#         'State', ['passengers', 'taxi', 'buildings', 'outside']
#     )
#
#     class Region(object):
#         def __init__(self, tiles, subregions=None, superregion=None, i=None):
#             self.tiles = tiles
#             if subregions is None:
#                 subregions = []
#             self.subregions = []
#             for sr in subregions:
#                 if isinstance(sr, StructuredTaxiCab.RegionTuple):
#                     sr = StructuredTaxiCab.Region(**sr._asdict())
#                 elif isinstance(sr, StructuredTaxiCab.Region):
#                     pass
#                 self.subregions.append(sr)
#             self.superregion = superregion #this is a frozenset
#             self.i = i
#
#         def as_tuple(self):
#             subregion_tuples = []
#             for sr in self.subregions:
#                 subregion_tuples.append(sr)
#             subregion_tuples = tuple(subregion_tuples)
#             return StructuredTaxiCab.RegionTuple(
#                 self.tiles,
#                 subregion_tuples,
#                 self.superregion,
#                 self.i)
#
#     class State(object):
#         entity_list = ['passengers', 'taxi', 'buildings', 'outside']
#
#         def entities_specified(self, state_tuple):
#             for entity in StructuredTaxiCab.State.entity_list:
#                 if not hasattr(state_tuple, entity):
#                     return False
#             return True
#
#         def __init__(self,
#                      passengers=None,
#                      taxi=None,
#                      buildings=None,
#                      outside=None,
#                      state_tuple=None):
#             if state_tuple is not None and self.entities_specified(state_tuple):
#                 passengers = []
#                 for ptup in state_tuple.passengers:
#                     p = TaxiCabMDP.Passenger(**ptup._asdict())
#                     passengers.append(p)
#                 ttup = state_tuple.taxi
#                 taxi = TaxiCabMDP.Taxi(**ttup._asdict())
#                 buildings = []
#                 for btup in state_tuple.buildings:
#                     b = StructuredTaxiCab.Region(**btup._asdict())
#                     buildings.append(b)
#                 otup = state_tuple.outside
#                 outside = StructuredTaxiCab.Region(**otup._asdict())
#             self.passengers = passengers
#             self.taxi = taxi
#             self.buildings = buildings
#             self.outside = outside
#
#         def as_tuple(self):
#             return StructuredTaxiCab.StateTuple(
#                 passengers=tuple([p.as_tuple() for p in self.passengers]),
#                 taxi=self.taxi.as_tuple(),
#                 buildings=tuple([b.as_tuple() for b in self.buildings]),
#                 outside=self.outside.as_tuple()
#             )
#
#     # =============== #
#     #  MDP Interface  #
#     # =============== #
#     def __init__(self,
#                  width=6, height=6,
#                  walls=None,
#                  building_params: "walls defined by corners "
#                                   "with exit/entrances" = None,
#                  locations=None,
#                  init_location=(0, 3),
#                  init_passengers=None,
#                  step_cost=-1
#                  ):
#         if building_params is not None:
#             if walls is None:
#                 walls = []
#             for bparams in building_params:
#                 bwalls = get_building_walls(**bparams)
#                 walls.extend(bwalls)
#         else:
#             building_params = []
#         self.building_params = building_params
#
#         TaxiCabMDP.__init__(self, width=width, height=height, walls=walls,
#                             locations=locations, init_location=init_location,
#                             init_passengers=init_passengers,
#                             step_cost=step_cost)
#
#     def transition_reward_dist(self, s=None, a=None):
#         if a == self.TERMINAL_ACTION:
#             return {(self.TERMINAL_STATE, 0): 1}
#         if s is None:
#             s = self.last_state
#         else:
#             s = StructuredTaxiCab.State(state_tuple=s)
#         s, r = self._update_state_get_reward(s, a)
#         self.last_state = s
#         return {(s.as_tuple(), r): 1}
#
#     def get_init_state(self):
#         passengers = [TaxiCabMDP.Passenger(**p) for p in self.init_passengers]
#         taxi = TaxiCabMDP.Taxi(self.init_location, gas=100)
#         buildings = []
#         outside = set(product(range(self.width), range(self.height)))
#         locations = set(self.locs)
#         for bi, bp in enumerate(self.building_params):
#             interior_tiles = frozenset(get_interior_tiles(bp['corners']))
#
#             #get locations in this building
#             locs = list(locations.intersection(interior_tiles))
#             loc_regions = []
#             for l in locs:
#                 # loc_region = StructuredTaxiCab.Region(
#                 #     tiles=(l,), superregion=frozenset(interior_tiles)
#                 # )
#                 loc_regions.append((l,))
#
#             b = StructuredTaxiCab.Region(tiles=interior_tiles,
#                                          subregions=loc_regions,
#                                          i=bi)
#             buildings.append(b)
#             outside.difference_update(interior_tiles)
#
#         #get locations outside
#         locs = list(locations.intersection(outside))
#         loc_regions = []
#         for l in locs:
#             # loc_region = StructuredTaxiCab.Region(
#             #     tiles=(l,), superregion=frozenset(outside)
#             # )
#             loc_regions.append((l,))
#
#         outside = StructuredTaxiCab.Region(tiles=frozenset(outside),
#                                            subregions=loc_regions)
#         self.last_state = StructuredTaxiCab.State(
#             passengers, taxi, buildings, outside
#         )
#         return self.last_state.as_tuple()