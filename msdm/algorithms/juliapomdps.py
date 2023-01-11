from msdm.core.pomdp import TabularPOMDP
from msdm.core.algorithmclasses import Learns, Result
import importlib
import numpy as np

class JuliaPOMDP(Learns):
    def __init__(self, solvername, solvermodule, *args, **kwargs):
        import julia
        try:
            m = getattr(importlib.import_module(solvermodule), solvername)
            # TODO; how to incorporate optional arguments in a generic way?
            self.solver = m()
        except (ImportError, julia.core.UnsupportedPythonError):
            raise Exception("Before using Julia POMDPs, must install with\n\tpython -m msdm.tools.install_julia_pomdps")

    def train_on(self, pomdp: TabularPOMDP):
        # We intentionally avoid importing at the top-level to ensure we avoid
        # errors from users without julia.
        from julia.QuickPOMDPs import DiscreteExplicitPOMDP
        from julia.POMDPPolicies import alphavectors
        from julia.POMDPs import solve

        m = DiscreteExplicitPOMDP(
            pomdp.state_list,
            pomdp.action_list,
            pomdp.observation_list,
            lambda s, a, ns: pomdp.next_state_dist(s, a).prob(ns),
            lambda a, ns, o: pomdp.observation_dist(a, ns).prob(o),
            lambda s, a: pomdp.next_state_dist(s, a).expectation(lambda ns: pomdp.reward(s, a, ns)),
            pomdp.discount_rate
        )

        policy = solve(self.solver, m)

        from julia.Main import typeof as typeof2
        alphas = np.array([np.array(v) for v in alphavectors(policy)])

        def value(belief):
            bvec = np.array([belief.prob(s) for s in pomdp.state_list])
            assert alphas.shape[1] == len(bvec)
            return np.max(alphas @ bvec)

        return Result(
            valuefn=value,
            juliapolicy=policy,
            alphavectors=alphas,
        )

    @classmethod
    def QMDP(cls, *args, **kwargs):
        return cls('QMDPSolver', 'julia.QMDP', *args, **kwargs)

    @classmethod
    def IncrementalPruning(cls, *args, **kwargs):
        return cls('PruneSolver', 'julia.IncrementalPruning', *args, **kwargs)

    @classmethod
    def SARSOP(cls, *args, **kwargs):
        return cls('SARSOPSolver', 'julia.SARSOP', *args, **kwargs)

    @classmethod
    def ARDESPOT(cls, *args, **kwargs):
        return cls('DESPOTSolver', 'julia.ARDESPOT', *args, **kwargs)

if __name__ == '__main__':
    from msdm.core.distributions import DictDistribution
    from msdm.domains.tiger import Tiger
    mdp = Tiger(0.85, discount_rate=0.95)
    res = JuliaPOMDP.QMDP().train_on(mdp)
    print(res)
    print('VALUE', res.valuefn(mdp.initial_state_dist()))
