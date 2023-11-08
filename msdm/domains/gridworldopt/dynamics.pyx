import cython
from libc.math cimport sin, cos, pi

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void transition_reward_matrices(
    double step_cost, 
    double wall_bump_cost,
    double stay_prob,
    double left_slip_prob,
    double right_slip_prob,
    double back_slip_prob,
    int [:, :] state_list, # x, y
    int [:, :] action_list, # dx, dy
    int [:, :] xy_walls, # x, y
    double [:, :] xy_rewards, # x, y
    double [:, :, :, :, :] transition_matrix, # x, y, ai, nx, ny
    double [:, :, :] reward_matrix, # x, y, ai
):
    cdef int si, ai, x, y, dx, dy
    cdef FixedParams params = FixedParams(
        width=xy_walls.shape[0],
        height=xy_walls.shape[1],
        n_states=state_list.shape[0],
        n_actions=action_list.shape[0],
        step_cost=step_cost,
        wall_bump_cost=wall_bump_cost,
        step_cost=step_cost,
        stay_prob=stay_prob,
        left_slip_prob=left_slip_prob,
        right_slip_prob=right_slip_prob,
        back_slip_prob=back_slip_prob,
        forward_prob = 1 - left_slip_prob - right_slip_prob - back_slip_prob - stay_prob
    )
    for si in range(params.n_states):
        for ai in range(params.n_actions):
            x = state_list[si, 0]
            y = state_list[si, 1]
            dx = action_list[ai, 0]
            dy = action_list[ai, 1]
            transition_reward(
                x=x, y=y, dx=dx, dy=dy,
                xy_walls=xy_walls,
                xy_rewards=xy_rewards,
                next_state_dist=transition_matrix[x, y, ai],
                reward_item=reward_matrix[x, y, ai:ai+1],
                params=params, 
            )

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void transition_reward(
    int x, int y, int dx, int dy,
    int [:, :] xy_walls,
    double [:, :] xy_rewards,
    double [:, :] next_state_dist, 
    double [:] reward_item, 
    FixedParams params,
):
    cdef int nx, ny, dx_, dy_
    cdef double expected_reward = 0.

    # staying in place
    if params.stay_prob > 0:
        next_state_dist[x, y] += params.stay_prob
        expected_reward += xy_rewards[x, y]*params.stay_prob

    # moving forward
    if params.forward_prob > 0:
        nx, ny = next_state(x, y, dx, dy, xy_walls, params)
        next_state_dist[nx, ny] += params.forward_prob
        if nx == x and ny == y:
            expected_reward += params.wall_bump_cost*params.forward_prob
        expected_reward += xy_rewards[nx, ny]*params.forward_prob

    # slipping left
    if params.left_slip_prob > 0:
        dx_, dy_ = rotate_left(dx, dy)
        nx, ny = next_state(x, y, dx_, dy_, xy_walls, params)
        next_state_dist[nx, ny] += params.left_slip_prob
        if nx == x and ny == y:
            expected_reward += params.wall_bump_cost*params.left_slip_prob
        expected_reward += xy_rewards[nx, ny]*params.left_slip_prob
    
    # slipping right
    if params.right_slip_prob > 0:
        dx_, dy_ = rotate_right(dx, dy)
        nx, ny = next_state(x, y, dx_, dy_, xy_walls, params)
        next_state_dist[nx, ny] += params.right_slip_prob
        if nx == x and ny == y:
            expected_reward += params.wall_bump_cost*params.right_slip_prob
        expected_reward += xy_rewards[nx, ny]*params.right_slip_prob
    
    # slipping back
    if params.back_slip_prob > 0:
        dx_, dy_ = rotate_back(dx, dy)
        nx, ny = next_state(x, y, dx_, dy_, xy_walls, params)
        next_state_dist[nx, ny] += params.back_slip_prob
        if nx == x and ny == y:
            expected_reward += params.wall_bump_cost*params.back_slip_prob
        expected_reward += xy_rewards[nx, ny]*params.back_slip_prob

    # state-action reward is the expected reward
    reward_item[0] = expected_reward + params.step_cost

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (int, int) next_state(int x, int y, int dx, int dy, int [:, :] xy_walls, FixedParams params):
    cdef int nx = x + dx
    cdef int ny = y + dy
    if not in_bounds(nx, ny, params) or xy_walls[nx, ny]:
        nx = x
        ny = y
    return nx, ny 

cdef bint in_bounds(int x, int y, FixedParams params):
    return 0 <= x < params.width and 0 <= y < params.height

cdef (double, double) rotate(double dx, double dy, double rad):
    cdef double ndx = dx*cos(rad) - dy*sin(rad)
    cdef double ndy = dx*sin(rad) + dy*cos(rad)
    return ndx, ndy

cdef (int, int) rotate_left(int dx, int dy):
    cdef double nx, ny 
    nx, ny = rotate(dx, dy, pi/2)
    return int(nx), int(ny)

cdef (int, int) rotate_right(int dx, int dy):
    cdef double nx, ny
    nx, ny = rotate(dx, dy, -pi/2)
    return int(nx), int(ny)

cdef (int, int) rotate_back(int dx, int dy):
    cdef double nx, ny
    nx, ny = rotate(dx, dy, pi)
    return int(nx), int(ny)

cdef struct FixedParams:
    int height
    int width
    int n_states
    int n_actions
    double step_cost
    double wall_bump_cost
    double stay_prob
    double left_slip_prob
    double right_slip_prob
    double back_slip_prob
    double forward_prob
