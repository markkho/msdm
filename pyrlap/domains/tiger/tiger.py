from pyrlap.core.pomdp import PartiallyObservableMarkovDecisionProcess
from pyrlap.core.mdp import DictionaryMDP

class TigerProblem(PartiallyObservableMarkovDecisionProcess):
    def __init__(self,
                 listen_cost=-1,
                 tiger_cost=-100,
                 notiger_reward=10,
                 roar_prob=.9
                 ):
        mdp = DictionaryMDP(
            transition_dict={
                'left-tiger': {
                    'listen': {'left-tiger': 1.0},
                    'left-door': {
                        'left-tiger': 0.5,
                        'right-tiger': 0.5
                    },
                    'right-door': {
                        'left-tiger': 0.5,
                        'right-tiger': 0.5
                    }
                },
                'right-tiger': {
                    'listen': {'right-tiger': 1.0},
                    'left-door': {
                        'left-tiger': 0.5,
                        'right-tiger': 0.5
                    },
                    'right-door': {
                        'left-tiger': 0.5,
                        'right-tiger': 0.5
                    }
                }
            },
            reward_dict={
                'left-tiger': {
                    'listen': {'left-tiger': listen_cost},
                    'left-door': {
                        'left-tiger': tiger_cost,
                        'right-tiger': tiger_cost
                    },
                    'right-door': {
                        'left-tiger': notiger_reward,
                        'right-tiger': notiger_reward
                    }
                },
                'right-tiger': {
                    'listen': {'right-tiger': listen_cost},
                    'left-door': {
                        'left-tiger': notiger_reward,
                        'right-tiger': notiger_reward
                    },
                    'right-door': {
                        'left-tiger': tiger_cost,
                        'right-tiger': tiger_cost
                    }
                }
            },
            init_state_dist={
                'left-tiger': .5,
                'right-tiger': .5
            }
        )
        super().__init__(mdp)
        self.roar_prob = roar_prob

    def observation_dist(self, a, ns):
        if a == 'listen':
            if ns == 'left-tiger':
                return {
                    'left-roar': self.roar_prob,
                    'right-roar': 1 - self.roar_prob
                }
            elif ns == 'right-tiger':
                return {
                    'left-roar': 1 - self.roar_prob,
                    'right-roar': self.roar_prob
                }
        else:
            return {'reset': 1.0}

    def available_actions(self):
        return ['listen', 'left-door', 'right-door']