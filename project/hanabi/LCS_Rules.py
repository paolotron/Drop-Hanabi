from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from numpy import uint8


@dataclass
class RuleSet:
    match_string: NDArray[bool]
    dont_care: NDArray[bool]
    action: NDArray[bool]

    @staticmethod
    def unpack_rules(rule_array: NDArray[uint8], rule_length: int):
        """
        Create RuleSet from packaged bitstring, the format for a single rule is
        [rule matching bits | don't care bits | action bits]
        each of this 3 section is padded to 8 bits
        a complete RuleSet is composed of a matrix of rules
        @param rule_array: BitMatrix
        @param rule_length: length of the rules
        @return: RuleSet
        """
        match_string = rule_array[:, :rule_length].copy()
        dont_care = rule_array[:, rule_length:2 * rule_length].copy()
        action = rule_array[:, 2 * rule_length:].copy()
        return RuleSet(match_string,
                       dont_care,
                       action)

    @staticmethod
    def empty_rules(rule_length: int, action_length: int):
        """
        Create a RuleSet with a single random rule
        @param rule_length: length of the rules
        @param action_length: length of the actions
        @return: RuleSet
        """
        return RuleSet(np.ndarray(shape=(1, rule_length), dtype=bool),
                       np.ndarray(shape=(1, rule_length), dtype=bool),
                       np.ndarray(shape=(1, action_length), dtype=bool),)

    @staticmethod
    def random_rule_set(rule_length: int, action_length: int, number_of_rules: int):
        """
        Create a Random Rule Set in the specified shape
        @param rule_length: int
        @param action_length: int
        @param number_of_rules: ibt
        @return: RuleSet
        """
        dont_care = np.random.choice(a=[False, True], size=(number_of_rules, rule_length))
        match_string = np.random.choice(a=[False, True], size=(number_of_rules, rule_length))
        action_string = np.random.choice(a=[False, True], size=(number_of_rules, action_length))
        return RuleSet(dont_care, match_string, action_string)

    @staticmethod
    def create_rule_set(rule_match: NDArray[bool], dont_care: NDArray[bool], action: NDArray[bool]):
        assert rule_match.shape == dont_care.shape
        assert rule_match.shape[0] == action.shape[0]
        return RuleSet(rule_match, dont_care, action)

    def sensor_length(self):
        return self.match_string.shape[1]

    def action_length(self):
        return self.action.shape[1]

    def number_rules(self):
        return self.match_string.shape[0]

    def cover(self, situation: NDArray[uint8]) -> None:
        """
        Cover for an unknown event
        @param situation: sensor array
        @return: None
        """
        dont_care = np.random.choice(a=[False, True], size=(1, self.match_string.shape[1]))
        action = np.random.choice(a=[False, True], size=(1, self.action.shape[1]))
        self.match_string = np.vstack([self.match_string, situation])
        self.dont_care = np.vstack([self.dont_care, dont_care])
        self.action = np.vstack([self.action, action])

    def match(self, situation: NDArray[uint8]) -> NDArray[np.bool_]:
        """
        Match sensor array to
        @param situation: sensor array
        @return: bool array of matched rules
        """
        return np.all((self.dont_care | (~(self.match_string ^ situation))), axis=1)

    def get_action(self, index: int) -> NDArray[uint8]:
        """
        Get action from rule index
        @param index: index of the rule got from match string
        @return: action bitstring
        """
        return self.action[index]

    def pack_rules(self):
        """
        Return compressed representation of RuleSet with the following format
        [rule matching bits | don't care bits | action bits]
        each of this 3 section is padded to 8 bits
        a complete RuleSet is composed of a matrix of rules
        @return: NDarray
        """
        return np.hstack([self.match_string, self.dont_care, self.action])


@dataclass(repr=False, frozen=True)
class EndResult:
    """
    @rule_match: bool matrix with n_rules columns and n_turns rows, True if rule was matcherd
    @critical_rules: array with n_rules size, True if rule casued fatal crash
    @rule_usage: list ints that list the indexes of activated rules
    """
    rule_match: NDArray
    critical_rules: NDArray
    rule_usage: Tuple[int]


class LCSRules:
    """
    Learning Classifier System actor:
    implements the matching and covering phase
    """

    def __init__(self, sensor_list: List, action_length: int, rule: RuleSet = None):
        """
        Initialization method
        @param sensor_list: List of Sensors that must inherit from GenericSensor,
        they will be used to decode the environment
        @param rule: RuleSet that will be used for the actions
        @param action_length: Length of the action string
        """
        self.__sensor_list = sensor_list
        self.__sensor_size = sum(map(lambda x: x.get_out_size(), sensor_list))
        if rule is None:
            self.__rule = RuleSet.empty_rules(self.__sensor_size, action_length)
        else:
            self.__rule = rule
        self.__rule_use = []
        self.__rule_match = np.zeros((1, self.__rule.action.shape[0]), dtype=bool)
        self.__critical_rules = np.zeros(self.__rule.action.shape[0], dtype=bool)
        self.__action_length = action_length

    def act(self, environment) -> NDArray[np.bool_]:
        """
        Method to get an action from the current position
        @param environment: Input of Activate Method from Sensors
        @return: np.ndarray[bool]
        """
        sensor_list = self.__detect(environment)
        sensor_activation = np.hstack(sensor_list)
        rule_activation = self.__match(sensor_activation)
        rule_activation *= ~self.__critical_rules
        self.__rule_match = np.vstack([self.__rule_match, rule_activation.reshape(1, -1)])
        if np.sum(rule_activation) == 0:
            self.__cover(sensor_activation)
            rule_activation = self.__match(sensor_activation)
            self.__rule_match = np.vstack([self.__rule_match, rule_activation])
            rule_activation *= ~self.__critical_rules

        activated_rules = np.argwhere(rule_activation).reshape(-1,)
        # Choose the more specific rule
        choice = activated_rules[np.argmin(np.sum(self.__rule.dont_care[activated_rules], axis=1))]

        self.__rule_use.append(choice)
        action = self.__rule.get_action(choice - 1)

        return action

    def end_game_data(self) -> EndResult:
        """
        Post-mortem data of usage of the rules,
        returns EndResult struct
        """
        return EndResult(self.__rule_match, self.__critical_rules, self.__rule_use)

    def signal_critical_failure(self):
        self.__critical_rules[self.__rule_use[-1]] = True

    def get_rule_set(self) -> RuleSet:
        """
        Get the ruleset
        @return: RuleSet
        """
        return self.__rule

    def __detect(self, environment) -> List[NDArray[uint8]]:
        """
        Return Sensor output in bitstring format
        @param environment: input of Sensors
        @return: np.ndarray
        """
        actions = []
        for sensor in self.__sensor_list:
            actions.append(sensor.get_activate(environment).astype(np.bool_))
        return actions

    def __match(self, sensor_activation: NDArray[uint8]) -> NDArray[uint8]:
        """
        return matched rules
        @param sensor_activation: bitstring
        @return: bitstring
        """
        return self.__rule.match(sensor_activation)

    def __cover(self, sensor_activation: NDArray[uint8]):
        """
        add a new random to cover for an unmatched situation
        @param sensor_activation:
        @return:
        """
        self.__rule.cover(sensor_activation)
        self.__rule_match = np.hstack([self.__rule_match, np.zeros((self.__rule_match.shape[0], 1))])
        self.__critical_rules = np.append(self.__critical_rules, False)
        # self.__rule_use.append(1)
        # self.__rule_match = np.append(self.__rule_use, 1)
        return len(self.__rule_match) - 1
