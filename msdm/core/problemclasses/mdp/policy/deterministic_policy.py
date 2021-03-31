class DeterministicPolicy:
    def action(self, s):
        ad = self.action_dist(s)
        assert len(ad.support) == 1
        return next(iter(ad.support))
