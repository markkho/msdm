import warnings
warnings.warn("officerobot.py is still experimental.")

from msdm.core.distributions.jointprobabilitytable import \
    Assignment, JointProbabilityTable
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
import functools

movement_prob = .99
movement_list = ["clockwise", "counter_clockwise", "stay"]


coffee_request_prob = .2
coffee_request_persistence = .99
has_coffee_persistence = .99
get_coffee_prob = .999999
deliver_coffee_prob = .999999

mail_waiting_prob = .1
mail_waiting_persistence = .99
has_mail_persistence = .99
get_mail_prob = .999999
deliver_mail_prob = .999999

clean_prob = .99
messier_prob = .1
mess_remain_prob = .89
tidy_levels = [0, 1, 2, 3, 4, 5]

def lru_cache(maxsize):
    def wrapper(fn):
        cached = functools.lru_cache(maxsize=maxsize)(fn)
        wrapped = functools.wraps(fn)(cached)
        return wrapped
    return wrapper

@functools.lru_cache()
def movement_effect(location, movement) -> ["next_location"]:
    if movement == "stay":
        return JointProbabilityTable.from_pairs([
            [dict(next_location=location), 1.0],
        ])
    next_loc = {
        "office": {
            "counter_clockwise": "lab",
            "clockwise": "hallway",
        },
        "hallway": {
            "counter_clockwise": "office",
            "clockwise": "mailroom",
        },
        "mailroom": {
            "counter_clockwise": "hallway",
            "clockwise": "coffee_room",
        },
        "coffee_room": {
            "counter_clockwise": "mailroom",
            "clockwise": "lab",
        },
        "lab": {
            "counter_clockwise": "coffee_room",
            "clockwise": "office",
        }
    }[location][movement]
    return JointProbabilityTable.from_pairs([
        [dict(next_location=next_loc), movement_prob],
        [dict(next_location=location), 1 - movement_prob],
    ])

@functools.lru_cache()
def get_coffee_effect(location, has_coffee, get_coffee) -> ['next_has_coffee']:
    if location == "coffee_room" and get_coffee and not has_coffee:
        return JointProbabilityTable.from_pairs([
            [dict(next_has_coffee=True), get_coffee_prob],
            [dict(next_has_coffee=False), 1 - get_coffee_prob],
        ])

@functools.lru_cache()
def deliver_coffee_effect(location, has_coffee, coffee_request, deliver_coffee) -> ['next_coffee_request', 'next_has_coffee']:
    if deliver_coffee and has_coffee and location == "office" and coffee_request:
        return JointProbabilityTable.from_pairs([
            [dict(next_has_coffee=False, next_coffee_request=False), deliver_coffee_prob],
            [dict(next_has_coffee=True, next_coffee_request=True), 1 - deliver_coffee_prob],
        ])

@functools.lru_cache()
def has_coffee_effect(has_coffee) -> ['next_has_coffee']:
    return JointProbabilityTable.from_pairs([
        [dict(next_has_coffee=has_coffee), has_coffee_persistence],
        [dict(next_has_coffee=(not has_coffee)), 1 - has_coffee_persistence],
    ])

@functools.lru_cache()
def coffee_request_effect(coffee_request) -> ['next_coffee_request']:
    if not coffee_request:
        return JointProbabilityTable.from_pairs([
            [dict(next_coffee_request=True), coffee_request_prob],
            [dict(next_coffee_request=False), 1 - coffee_request_prob]
        ])
    return JointProbabilityTable.from_pairs([
        [dict(next_coffee_request=True), coffee_request_persistence],
        [dict(next_coffee_request=False), 1 - coffee_request_persistence]
    ])

@functools.lru_cache()
def cleaning_effect(location, tidiness, clean) -> ["next_tidiness"]:
    if clean and location == "lab":
        t = JointProbabilityTable.from_pairs([
            [dict(next_tidiness=min(max(tidy_levels), tidiness+1)), clean_prob],
            *[
                [dict(next_tidiness=i), (1-clean_prob)/len(tidy_levels)]
                for i in tidy_levels
            ]
        ])
        t._check_valid()
        return t

@functools.lru_cache()
def mess_effect(tidiness) -> ["next_tidiness"]:
    rand_prob = 1 - mess_remain_prob - messier_prob
    return JointProbabilityTable.from_pairs([
        [dict(next_tidiness=max(min(tidy_levels), tidiness-1)), messier_prob],
        [dict(next_tidiness=tidiness), mess_remain_prob],
        *[
            [dict(next_tidiness=i), rand_prob/len(tidy_levels)]
            for i in tidy_levels
        ]
    ])

@functools.lru_cache()
def get_mail_effect(location, mail_waiting, get_mail, has_mail) -> ["next_mail_waiting", "next_has_mail"]:
    if get_mail and not has_mail and location == "mailroom" and mail_waiting:
        return JointProbabilityTable.from_pairs([
            [dict(next_mail_waiting=False, next_has_mail=True), get_mail_prob],
            [dict(next_mail_waiting=True, next_has_mail=False), 1 - get_mail_prob],
        ])

@functools.lru_cache()
def deliver_mail_effect(location, has_mail, deliver_mail) -> ["next_has_mail"]:
    if deliver_mail and has_mail and location == "office":
        return JointProbabilityTable.from_pairs([
            [dict(has_mail=False), deliver_mail_prob],
            [dict(has_mail=False), 1 - deliver_mail_prob],
        ])

@functools.lru_cache()
def has_mail_effect(has_mail) -> ['next_has_mail']:
    return JointProbabilityTable.from_pairs([
        [dict(next_has_mail=has_mail), has_mail_persistence],
        [dict(next_has_mail=(not has_mail)), 1 - has_mail_persistence],
    ])

@functools.lru_cache()
def mail_arrival_effect(mail_waiting) -> ["next_mail_waiting"]:
    if not mail_waiting:
        return JointProbabilityTable.from_pairs([
            [dict(next_mail_waiting=True), mail_waiting_prob],
            [dict(next_mail_waiting=False), 1 - mail_waiting_prob]
        ])
    return JointProbabilityTable.from_pairs([
        [dict(next_mail_waiting=True), mail_waiting_persistence],
        [dict(next_mail_waiting=False), 1 - mail_waiting_persistence]
    ])

def next_state_dist(s, a):
    dist = JointProbabilityTable.from_pairs([
        [s+a, 1.0]
    ])
    move_dist = dist.then(movement_effect)

    clean_dist = dist.then(cleaning_effect)
    clean_dist = clean_dist.then(mess_effect)

    coffee_dist = dist.then(has_coffee_effect)
    coffee_dist = coffee_dist.then(get_coffee_effect)
    coffee_dist = coffee_dist.then(deliver_coffee_effect)
    coffee_dist = coffee_dist.then(coffee_request_effect)

#     mail_dist = dist.then(has_mail_effect)
#     mail_dist = mail_dist.then(get_mail_effect)
#     mail_dist = mail_dist.then(mail_arrival_effect)
#     mail_dist = mail_dist.then(deliver_mail_effect)

    dist = move_dist.join(clean_dist).join(coffee_dist)
#     .join(mail_dist)
    ns_dist = dist.normalize().groupby(lambda c: "next_" in c)
    return ns_dist.rename_columns(lambda c: c.replace("next_", ""))

def initial_state_dist():
    return JointProbabilityTable.deterministic(
        dict(
            location="coffee_room",
            coffee_request=True,
            has_coffee=False,
            has_mail=False,
            tidiness=2,
            mail_waiting=False
        )
    )

def reward(s, a, ns):
    r = 0
    r += -5 if s['coffee_request'] else 0
    r += -.1*(max(tidy_levels) - s['tidiness']) if s['tidiness'] is not None else 0
    r += -1 if s['mail_waiting'] or s['holding'] == 'mail' else 0
    r += -.1 if a['clean'] else 0
    return r

def make_action_list():
    default_values = dict(
        movement="stay",
        get_coffee=False,
        clean=False,
        get_mail=False,
        deliver_mail=False,
        deliver_coffee=False
    )
    action_params = dict(
        movement=movement_list,
        clean=[True, False],
        get_coffee=[True, False],
        deliver_coffee=[True, False],
        get_mail=[True, False],
        deliver_mail=[True, False],
    )
    action_list = [Assignment.from_kwargs(**default_values)]
    for variable, values in action_params.items():
        for value in values:
            kwargs = {**default_values, variable: value}
            action = Assignment.from_kwargs(**kwargs)
            if action not in action_list:
                action_list.append(action)
    return tuple(action_list)

action_list = make_action_list()

class OfficeRobot(TabularMarkovDecisionProcess):
    """
    Office robot example from Boutilier, Dean & Hanks (1999)
    Decision-Theoretic Planning: Structural Assumptions and
    Computational Leverage.
    """
    discount_rate = .95
    def next_state_dist(self, s, a):
        return next_state_dist(s, a)

    def reward(self, s, a, ns):
        return reward(s, a, ns)

    def actions(self, s):
        return action_list

    def initial_state_dist(self):
        return initial_state_dist()

    def is_terminal(self, s):
        return False
