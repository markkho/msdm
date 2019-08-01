import torch
from pyrlap.core.agent import Planner

def softmax(vals, temp, min_energy=.01):
    norm_vals = (vals.t() - torch.max(vals, dim=1)[0]).t()
    energy = torch.exp((norm_vals.t()*temp).t()) + torch.tensor(min_energy)
    norm_energy = energy.sum(dim=1)
    return (energy.t()/norm_energy).t()

class SoftValueIteration(Planner):
    def __init__(self, mdp,
                 discount_rate=.99,
                 converge_delta=.0001,
                 max_iterations=100,
                 softmax_temp=0.0,
                 randchoose=0.0,
                 init_val=0.0):
        Planner.__init__(self, mdp)
        self.discount_rate = discount_rate
        self.converge_delta = converge_delta
        self.max_iterations = max_iterations
        self.softmax_temp = softmax_temp
        self.randchoose = randchoose
        self.init_val = init_val
        self.device = 'cpu'

    def solve(self):
        mats = self.mdp.as_matrices()
        tf = torch.from_numpy(mats['tf']).to(self.device)
        rf = torch.from_numpy(mats['rf']).to(self.device)

        er = torch.einsum("san,san->sa", tf, rf)
        q = torch.zeros(tf.shape[0:2])
        for i in range(self.max_iterations):
            # next step discounted softmax value
            s_softval = torch.einsum("sa->s", softmax(q, self.softmax_temp) * q)
            disc_ns_softval = self.discount_rate * s_softval

            # future state-action value
            fq = torch.einsum("san,n->sa", tf, disc_ns_softval)

            diff = torch.abs((er + fq) - q).max()
            if diff < self.converge_delta:
                break
            # sa-val is the expected r + discounted future value
            q = er + fq
        pol = softmax(q, self.softmax_temp)
        v = torch.einsum("sa->s", pol * q)
        pol = pol.data.numpy()
        v = v.data.numpy()
        q = q.data.numpy()
        self.optimal_policy = \
            {s: dict(zip(mats['aa'], adist)) for s, adist in zip(mats['ss'], pol)}
        self.value_function = dict(zip(mats['ss'], v))
        self.action_value_function =\
            {s: dict(zip(mats['aa'], aq)) for s, aq in zip(mats['ss'], q)}