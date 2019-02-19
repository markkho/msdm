# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

import time

import numpy as np
import pandas as pd
import seaborn as sns

from pyrlap.domains.gridworld import GridWorld
from pyrlap.algorithms.qlearning import Qlearning
from pyrlap.domains.gridworld.gridworldvis import visualize_trajectory

# %%
gw = GridWorld(
    gridworld_array=['...........',
                     '.xxxxxxxxxy',
                     '.xxxxxxxxxx'],
    absorbing_states=[(10, 1),],
    init_state=(0, 1),
    feature_rewards={'.':-1, 'x':-10, 'y':100})
s_features = gw.state_features

# %%
np.random.seed(1234)
all_run_data = []

# %%
start = time.time()
for i in range(20):
    params = {'learning_rate': 1,
              'eligibility_trace_decay': .8,
              'initial_qvalue': 100}
    qlearn = Qlearning(gw, 
                       softmax_temp=.2, 
                       discount_rate=.99,
                       **params)
    run_data = qlearn.train(episodes=50, 
                            max_steps=100,
                            run_id=i,
                            return_run_data=True)
    for r in run_data:
        r.update(params)
    all_run_data.extend(run_data)
print("total time: {:.2f}".format(time.time() - start))

# %%
run_df = pd.DataFrame(all_run_data)
run_df['is_x'] = run_df['s'].apply(lambda s: s_features[s] == 'x')
param_list = run_df[['learning_rate', 'initial_qvalue', 'eligibility_trace_decay']]
param_list = param_list.to_records(index=False)
param_list = [str(tuple(p)) for p in param_list]
run_df['params'] = param_list

# %%
ep_rewards = run_df.groupby(['run_id', 'episode', 'params'])['r']\
    .sum().reset_index()
ax = sns.pointplot(data=ep_rewards, 
                   x='episode', 
                   y='r', 
                   hue='params')

# %%
traj = qlearn.run(softmax_temp=0.0, randchoose=0.0)
gwp = gw.plot()
gwp.plot_trajectory(traj=[(s, a) for s, a, ns, r in traj])

# %%
