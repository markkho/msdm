import warnings
warnings.warn("officerobot.py is still experimental.")

import textwrap
from msdm.core.distributions.jointprobabilitytable import \
    Assignment, JointProbabilityTable
from msdm.core.distributions.factors import make_factor
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp.factoredmdp import FactoredMDP

class OfficeRobot(FactoredMDP,TabularMarkovDecisionProcess):
    """
    Office robot example from Boutilier, Dean & Hanks (1999)
    Decision-Theoretic Planning: Structural Assumptions and
    Computational Leverage.
    """
    def __init__(
        self,
        discount_rate=.95,
    ):
        self.discount_rate = discount_rate
        self.tidy_levels = (0, 1, 2, 3, 4, 5)
        self.movement_list = ("clockwise", "counter_clockwise", "stay")
        self.debug_mode = False

        super().__init__(
            initial_state_factors=self.make_initial_state_factors(),
            next_state_factors=self.make_next_state_factors(),
            reward_factors=self.make_reward_factors(),
            next_state_variable_substring="next_",
            reward_variable_substring="_reward",
        )

        self._action_list = self.make_action_list()

    def actions(self, s):
        return self._action_list

    def is_terminal(self, s):
        return False

    def state_string(self, s):
        template = textwrap.dedent(r"""
           .    ....office.-.....hallway
           .        /              \
           ....lab...        ....mailroom.
           .         \           /
           .      ....coffee_room
        """)
        robot = f"R{'c' if s['has_coffee'] else '_'}{'m' if s['has_mail'] else '_'}"
        office = f"office{'!' if s['coffee_request'] else ' '}"
        mailroom = f"mailroom{'!' if s['mail_waiting'] else ' '}"
        lab = f"lab({s['tidiness']})"
        state_string = template.\
            replace(f"....{s['location']}", f"{robot},{s['location']}").\
            replace(f"office.", office).\
            replace(f"mailroom.", mailroom).\
            replace(f"lab...", lab).\
            replace(".....hallway", "----.hallway").\
            replace(f".", ' ')
        return state_string

    def make_initial_state_factors(self):
        @make_factor(debug_mode=self.debug_mode)
        def initial_state_factor() -> [
            'location',
            'coffee_request',
            'has_coffee',
            'has_mail',
            'tidiness',
            'mail_waiting'
        ]:
            return JointProbabilityTable.deterministic(
                dict(
                    location="office",
                    coffee_request=False,
                    has_coffee=False,
                    has_mail=True,
                    tidiness=2,
                    mail_waiting=False
                )
            )
        return [
            initial_state_factor
        ]

    def make_next_state_factors(self):
        movement_prob = .99

        coffee_request_prob = .2
        coffee_request_persistence = .99
        has_coffee_persistence = .99
        get_coffee_prob = .999999
        deliver_coffee_prob = .999999

        mail_waiting_prob = .2
        mail_waiting_persistence = .99
        has_mail_persistence = .99
        get_mail_prob = .999999
        deliver_mail_prob = .999999

        clean_prob = .99
        messier_prob = .1
        mess_remain_prob = .89

        @make_factor(debug_mode=self.debug_mode)
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

        @make_factor(debug_mode=self.debug_mode)
        def get_coffee_effect(location, has_coffee, get_coffee) -> ['next_has_coffee']:
            if location == "coffee_room" and get_coffee and not has_coffee:
                return JointProbabilityTable.from_pairs([
                    [dict(next_has_coffee=True), get_coffee_prob],
                    [dict(next_has_coffee=False), 1 - get_coffee_prob],
                ])

        @make_factor(debug_mode=self.debug_mode)
        def deliver_coffee_effect(location, has_coffee, coffee_request, deliver_coffee) -> ['next_coffee_request', 'next_has_coffee']:
            if deliver_coffee and has_coffee and location == "office" and coffee_request:
                return JointProbabilityTable.from_pairs([
                    [dict(next_has_coffee=False, next_coffee_request=False), deliver_coffee_prob],
                    [dict(next_has_coffee=True, next_coffee_request=True), 1 - deliver_coffee_prob],
                ])

        @make_factor(debug_mode=self.debug_mode)
        def has_coffee_effect(has_coffee) -> ['next_has_coffee']:
            return JointProbabilityTable.from_pairs([
                [dict(next_has_coffee=has_coffee), has_coffee_persistence],
                [dict(next_has_coffee=(not has_coffee)), 1 - has_coffee_persistence],
            ])

        @make_factor(debug_mode=self.debug_mode)
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

        @make_factor(debug_mode=self.debug_mode)
        def cleaning_effect(location, tidiness, clean) -> ["next_tidiness"]:
            if clean and location == "lab":
                inc_tidiness = min(max(self.tidy_levels), tidiness+1)
                dec_tidiness = max(min(self.tidy_levels), tidiness-1)
                return JointProbabilityTable.from_pairs([
                    [dict(next_tidiness=inc_tidiness), clean_prob],
                    [dict(next_tidiness=tidiness), (1-clean_prob)/2],
                    [dict(next_tidiness=dec_tidiness), (1-clean_prob)/2],
                ])

        @make_factor(debug_mode=self.debug_mode)
        def mess_effect(tidiness) -> ["next_tidiness"]:
            inc_tidiness = min(max(self.tidy_levels), tidiness+1)
            dec_tidiness = max(min(self.tidy_levels), tidiness-1)
            return JointProbabilityTable.from_pairs([
                [dict(next_tidiness=dec_tidiness), messier_prob],
                [dict(next_tidiness=tidiness), mess_remain_prob],
                [dict(next_tidiness=inc_tidiness), 1 - mess_remain_prob - messier_prob],
            ])

        @make_factor(debug_mode=self.debug_mode)
        def get_mail_effect(location, mail_waiting, get_mail, has_mail) -> ["next_mail_waiting", "next_has_mail"]:
            if get_mail and not has_mail and location == "mailroom" and mail_waiting:
                return JointProbabilityTable.from_pairs([
                    [dict(next_mail_waiting=False, next_has_mail=True), get_mail_prob],
                    [dict(next_mail_waiting=True, next_has_mail=False), 1 - get_mail_prob],
                ])

        @make_factor(debug_mode=self.debug_mode)
        def deliver_mail_effect(location, has_mail, deliver_mail) -> ["next_has_mail"]:
            if deliver_mail and has_mail and location == "office":
                return JointProbabilityTable.from_pairs([
                    [dict(next_has_mail=False), deliver_mail_prob],
                    [dict(next_has_mail=True), 1-deliver_mail_prob],
                ])

        @make_factor(debug_mode=self.debug_mode)
        def has_mail_effect(has_mail) -> ['next_has_mail']:
            return JointProbabilityTable.from_pairs([
                [dict(next_has_mail=has_mail), has_mail_persistence],
                [dict(next_has_mail=(not has_mail)), 1 - has_mail_persistence],
            ])

        @make_factor(debug_mode=self.debug_mode)
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

        return [
            movement_effect,
            has_coffee_effect,
            get_coffee_effect,
            deliver_coffee_effect,
            coffee_request_effect,
            has_mail_effect,
            get_mail_effect,
            mail_arrival_effect,
            deliver_mail_effect,
            cleaning_effect,
            mess_effect
        ]

    def make_reward_factors(self):
        @make_factor(debug_mode=self.debug_mode)
        def calc_coffee_reward(coffee_request) -> ['coffee_reward']:
            return JointProbabilityTable.from_pairs([
                [dict(coffee_reward=-5*coffee_request), 1.0]
            ])

        @make_factor(debug_mode=self.debug_mode)
        def calc_tidiness_reward(tidiness) -> ['tidiness_reward']:
            return JointProbabilityTable.from_pairs([
                [dict(tidiness_reward=-1*(max(self.tidy_levels) - tidiness)), 1.0]
            ])

        @make_factor(debug_mode=self.debug_mode)
        def calc_mail_reward(mail_waiting, has_mail) -> ['mail_reward']:
            return JointProbabilityTable.from_pairs([
                [dict(mail_reward=-5*(mail_waiting or has_mail)), 1.0]
            ])

        @make_factor(debug_mode=self.debug_mode)
        def calc_cleaning_reward(clean) -> ['cleaning_reward']:
            return JointProbabilityTable.from_pairs([
                [dict(cleaning_reward=-1*clean), 1.0]
            ])

        return [
            calc_coffee_reward,
            calc_tidiness_reward,
            calc_mail_reward,
            calc_cleaning_reward
        ]

    def make_action_list(self):
        default_values = dict(
            movement="stay",
            get_coffee=False,
            clean=False,
            get_mail=False,
            deliver_mail=False,
            deliver_coffee=False
        )
        action_params = dict(
            movement=self.movement_list,
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
