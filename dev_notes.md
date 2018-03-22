# General structure

Goal is to have compositional MDPs and decision processes that can
be used for cog sci research.

An MDP is composed of a set of states, actions, transition function, initial state distribution, 
reward function (and discount rate):

$$
\mathcal{M} = <S, A, T, S_0, R, \gamma>
$$

Where both the transition function and reward function are stochastic: $T : S \times A \rightarrow \Pi(S)$ and $R : S \times A \times S \rightarrow \Pi(\mathbb{R})$. 

### __Default__ operations over elements of MDP

**State addition**, `S = S_1 + S_2`, is set union: $S = S_1 \cup S_2$. **State multiplication**, `S = S_1 * S_2`, is the Cartesian product of the sets: $S = S_1 \times S_2$.

Similarly, **action addition**, `A = A_1 + A_2`, is the set union for each state: $ A(s) = A_1(s) \cup A_2(s) $; **action multiplication** `A = A_1 * A_2` is the Cartesian product: $A(s) = A_1(s) \times A_2(s)$.

**Transition function addition**, `T = T_1 + T_2`, corresponds to $T(s, a)$ equalling $T_2(s, a)$ when $s \in S_2$ and $a \in A_2(s)$ and $T_2(s, a)$ otherwise. This is because $T_2$ "overrides" whatever $T_1$ says whenever it is relevant (namely, when it has something to say about state $s$ and action $a$).

- alternatively can use a "mixing" parameter $\epsilon$ for shared state/actions

<!--$$
T(s, a, s') = \begin{cases}
	T_2(s, a), & \text{if } s \in S_2 \land a \in A_2(s).\\
	T_1(s, a), & \text{otherwise}.
\end{cases}
$$-->

**Transition function multiplication**, `T = T_1 * T_2` corresponds to how models are typically composed. That is, $T((s_1', s_2') \mid (s_1, s_2), (a_1, a_2)) = T_1(s_1' \mid s_1, a_1)T_2(s_2' \mid s_2, a_2)$.

**Reward function addition** (`R = R_1 + R_2`): $R(s, a, s')$ equals $R_2(s, a, s')$ if $s \in S_2$, $a \in A_2(s)$, and $s' \in S_2$; otherwise, $R_1(s, a, s')$

- alternatively can use a weighting parameter for combining rewards in shared $s, a, s'$

**Reward function multiplication** `R = R_1 * R_2`: $R((s_1, s_2), (a_1, a_2), (s_1', s_2')) = (R_1(s_1, a_1, s_1'), R_2(s_2, a_2, s_2'))$. I.e. a reward vector is returned. 

### Features
- features can be over states, actions, nextstates, state-actions, or state-action-nextstates
- features can define transition-types or rewards





