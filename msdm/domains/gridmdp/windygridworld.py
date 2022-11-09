from collections import defaultdict
import matplotlib.pyplot as plt

from msdm.domains.gridmdp import GridMDP, Location, GridAction
from msdm.domains.gridmdp.plotting import GridMDPPlotter
from msdm.core.distributions import DictDistribution
from msdm.core.utils.funcutils import method_cache

# extension of DictDistribution
class DictDistribution(DictDistribution):
    @classmethod
    def from_tuples(cls, element_probs):
        dist = defaultdict(float)
        for e, p in element_probs:
            dist[e] += p
        return DictDistribution(dist)

    def normalize(self):
        total = sum(self.values())
        return DictDistribution({e: p/total for e, p in self.items()})

    def then(self, function):
        marg_then_dist = defaultdict(float)
        for e, p in self.items():
            then_dist = function(e)
            assert sum(then_dist.values()) == 1
            for then_e, then_p in then_dist.items():
                marg_then_dist[then_e] += p*then_p
        return DictDistribution(marg_then_dist)

class WindyGridWorld(GridMDP):
    def __init__(
        self,
        grid,
        feature_rewards=None,
        step_cost=-1,
        wall_bump_cost=-1,
        wind_probability=.5,
        discount_rate=.99,
        start_features="@",
        goal_features="$",
        wall_features="#",
    ):
        """
        In a windy gridworld, the wind can push the agent
        in a direction with some probability. By default, `v<^>`
        denotes wind in a certain direction,
        `@` denotes an agent start state, `#` denotes walls,
        and `$` denotes a goal state.
        """
        self.discount_rate = discount_rate
        self.wind_features = "^v<>"
        self.start_features = start_features
        self.goal_features = goal_features
        self.wall_features = wall_features
        self.step_cost = step_cost
        self.wall_bump_cost = wall_bump_cost
        self.wind_probability = wind_probability
        self.feature_rewards = feature_rewards
        super().__init__(grid)

    def initial_state_dist(self):
        initial_states = sum([self.locations_with(f) for f in self.start_features], ())
        return DictDistribution.uniform(initial_states)

    def next_state_dist(self, s, a):
        nsr_dist = self.next_state_reward_dist(s, a)
        return nsr_dist.marginalize(lambda nsr: nsr[0])

    def reward(self, s, a, ns):
        nsr_dist = self.next_state_reward_dist(s, a)
        nsr_dist = nsr_dist.condition(lambda nsr: nsr[0] == ns)
        return nsr_dist.expectation(lambda nsr: nsr[1])

    def is_absorbing(self, s):
        return self.feature_at(s) in self.goal_features

    @method_cache
    def next_state_reward_dist(self, s, a):
        nsr_dist = DictDistribution({(s, 0): 1})
        nsr_dist = nsr_dist.then(lambda nsr: self._effect_of_wind(*nsr))
        nsr_dist = nsr_dist.then(lambda nsr: self._effect_of_action(*nsr, a))
        nsr_dist = nsr_dist.then(lambda nsr: self._effect_of_walls(s, *nsr))
        nsr_dist = nsr_dist.then(lambda nsr: self._effect_of_features(*nsr))
        return nsr_dist

    def _effect_of_features(self, s : Location, r : float):
        f = self.feature_at(s)
        r += self.feature_rewards.get(f, 0.0)
        return DictDistribution({(s, r): 1})

    def _effect_of_wind(self, s : Location, r : float):
        f = self.feature_at(s)
        if f is None or f not in self.wind_features:
            return DictDistribution({(s, r): 1})
        if f == ">":
            ns = Location(s.x + 1, s.y)
        elif f == "<":
            ns = Location(s.x - 1, s.y)
        elif f == "^":
            ns = Location(s.x, s.y + 1)
        elif f == "v":
            ns = Location(s.x, s.y - 1)
        else:
            raise
        return DictDistribution({
            (s, r) : 1 - self.wind_probability,
            (ns, r): self.wind_probability,
        })

    def _effect_of_action(self, s : Location, r: float, a : GridAction):
        return DictDistribution({
            (Location(s.x + a.dx, s.y + a.dy), self.step_cost): 1
        })

    def _effect_of_walls(self, s : Location, ns : Location, r : float):
        if self.feature_at(ns) is not None and self.feature_at(ns) in self.wall_features:
            r += self.wall_bump_cost
            return DictDistribution({(s, r): 1})
        nx, ny = ns
        if ns.x < 0 or ns.x > (self.width - 1):
            r += self.wall_bump_cost
            nx = s.x
        if ns.y < 0 or ns.y > (self.height - 1):
            r += self.wall_bump_cost
            ny = s.y
        return DictDistribution({
            (Location(nx, ny), r) : 1
        })

    def state_string(self, s):
        chars = [list(r) for r in self._grid]
        for y, row in enumerate(chars):
            for x, f in enumerate(row):
                if f == '@':
                    chars[y][x] = '.'
                if Location(x, y) == s:
                    chars[y][x] = '@'
        return '\n'.join([''.join(r) for r in chars[::-1]])

    def plot(
        self,
        feature_colors=None,
        mark_initial_states=True,
        mark_absorbing_states=True,
        mark_wind=True,
        ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plotter = WindyGridWorldPlotter(self, ax=ax)
        if feature_colors is None:
            feature_colors = {
                **{d: 'lightgrey' for d in '<>v^'},
                'x': 'darkred',
                '#': 'k'
            }
        plotter.fill_features(
            feature_colors=feature_colors,
            default_color='w',
            Rectangle_kwargs=dict(
                zorder=-2
            )
        )
        if mark_initial_states:
            plotter.mark_features(
                feature_markers={'@': 'o'},
                plot_kwargs=dict(
                    markeredgecolor='cornflowerblue',
                    markersize=15,
                    markeredgewidth=2,
                    fillstyle='none',
                    zorder=-1
                )
            )
        if mark_absorbing_states:
            plotter.mark_features(
                feature_markers={'$': '*'},
                plot_kwargs=dict(
                    markeredgecolor='k',
                    markersize=25,
                    markeredgewidth=2,
                    color='yellow',
                    fillstyle='full',
                    zorder=-1
                )
            )
        if mark_wind:
            plotter.mark_features(
                feature_markers={
                    '<': 8,
                    '>': 9,
                    'v': 11,
                    '^': 10
                },
                plot_kwargs=dict(
                    markeredgecolor='darkgrey',
                    markersize=15,
                    markeredgewidth=2,
                    fillstyle='none',
                    zorder=-1
                )
            )
        plotter.plot_outer_box()
        return plotter

class WindyGridWorldPlotter(GridMDPPlotter):
    def plot_policy(
        self, policy, arrow_width=.1):
        return self.plot_location_action_map(
            location_action_map=policy,
            vmin=0,
            vmax=1,
            arrow_width=arrow_width,
            visualization_type='arrow',
            color_value_func=lambda v: 'k'
        )

if __name__ == "__main__":
    from msdm.algorithms.policyiteration import PolicyIteration
    wg = WindyGridWorld(
        grid="""
            ....$
            x^x<<
            x^x<<
            .^x<<
            x<<<<
            x<<<<
            x<<<<
            x<<<<
            x<<<<
            @....
        """,
        step_cost=-1,
        wall_bump_cost=-1,
        discount_rate=1.0,
        wind_probability=.5,
        feature_rewards={'x': -50, '$': 50}
    )
    pi_res = PolicyIteration().plan_on(wg)
    plotter = wg.plot()
    for _ in range(20):
        loc_traj = pi_res.policy.run_on(wg).state
        plotter.plot_location_trajectory(
            loc_traj,
            outline=True,
            main_PathPatch_kwargs=dict(
                edgecolor='green',
                linewidth=3,
                linestyle='--',
                zorder=10
            ),
        )
    plt.show()
