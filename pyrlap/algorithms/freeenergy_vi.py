import torch
from pyrlap.core.agent import Planner

def softmax(vals, temp, min_energy=.01):
    norm_vals = (vals.t() - torch.max(vals, dim=1)[0]).t()
    energy = torch.exp((norm_vals.t()*temp).t()) + torch.tensor(min_energy)
    norm_energy = energy.sum(dim=1)
    return (energy.t()/norm_energy).t()

class FreeEnergyValueIteration(Planner):
    """
    This solves for the policy that minimizes the
    free energy functional described by Eq. 39 and 41 in
    Ortega, Braun, Dyer, Kim & Tishby (2015).
    """
    def __init__(self, mdp,
                 discount_rate=.99,
                 converge_delta=.0001,
                 max_iterations=100,
                 softmax_temp=0.0,
                 randchoose=0.0,
                 default_policy=None,
                 info_cost_weight=.01,
                 init_val=0.0):
        Planner.__init__(self, mdp)
        self.discount_rate = discount_rate
        self.converge_delta = converge_delta
        self.max_iterations = max_iterations
        self.softmax_temp = softmax_temp
        self.randchoose = randchoose
        self.init_val = init_val
        self.device = 'cpu'
        self.default_policy = default_policy
        self.info_cost_weight = info_cost_weight

    def solve(self):
        mats = self.mdp.as_matrices()
        tf = torch.from_numpy(mats['tf']).to(self.device)
        rf = torch.from_numpy(mats['rf']).to(self.device)
        ss = mats['ss']
        aa = mats['aa']
        if self.default_policy is None:
            pi0 = torch.ones((len(ss), len(aa)), device=self.device)
            pi0 = torch.einsum("sa,s->sa", pi0, 1 / pi0.sum(dim=1))
        else:
            pi0 = torch.from_numpy(self.default_policy).to(self.device)

        er = torch.einsum("san,san->sa", tf, rf)
        fev = torch.zeros(len(ss))

        pi_eps = 1e-15 #tiny prob of taking any action prevents infinite cost

        for i in range(self.max_iterations):
            fut_fq = torch.einsum("san,n->sa",tf,fev)
            fq = er + self.discount_rate * fut_fq
            energy = (1 / self.info_cost_weight) * fq + torch.log(pi0)
            energy = (energy.t() - torch.max(energy, dim=1)[0]).t()
            pi = torch.exp(energy)
            pi = (1 - pi_eps)*pi + pi_eps/len(aa)
            z = pi.sum(dim=1)
            pi = torch.einsum("sa,s->sa",pi,1/z)
            new_fev = er - self.info_cost_weight*torch.log(pi/pi0) + \
                      self.discount_rate * fut_fq
            if (new_fev > 100).any():
                print(list(zip(ss, new_fev)))
                print(list(zip(ss, pi/pi0)))

            new_fev = torch.einsum("sa,sa->s", new_fev, pi)
            diff = torch.abs(new_fev - fev).max()
            if diff < self.converge_delta:
                break
            fev = new_fev
        fev = new_fev

        pol = pi.data.numpy()
        fev = fev.data.numpy()
        fq = fq.data.numpy()

        self.optimal_policy = \
            {s: dict(zip(mats['aa'], adist)) for s, adist in zip(mats['ss'], pol)}
        self.value_function = dict(zip(mats['ss'], fev))
        self.action_value_function =\
            {s: dict(zip(mats['aa'], aq)) for s, aq in zip(mats['ss'], fq)}
