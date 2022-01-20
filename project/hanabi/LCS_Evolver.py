from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Process
from threading import Thread
from typing import List

import numpy as np
from numpy.typing import NDArray
import LCS_Sensor as Sens
import server
from LCS_Actor import LCSActor
from LCS_Rules import LCSRules, EndResult
from LCS_Rules import RuleSet
from constants import *
from player_LCS import LCSPlayer


class Evolver:

    def __init__(self, num_players, pretrained: str = ""):
        self.num_players = num_players
        self.num_population = num_players
        self.gameManager = GameManager(num_players)
        if pretrained:
            rule = np.load(f"../hanabi/models/{pretrained}.npy")
            rule = RuleSet.unpack_rules(rule, self.gameManager.sensor_len())
            self.population = [rule.copy() for _ in range(self.num_population)]
        else:
            self.population = [RuleSet.empty_rules(self.gameManager.sensor_len(), self.gameManager.action_len())
                               for _ in range(self.num_population)]

    def evolve(self):
        results: List[Fitness] = tournament_play(self.population, self.gameManager, 10)[0]
        player = self.population[0]
        wins = list(filter(lambda x: not x.loss, results))
        if wins:
            fit = max(wins, key = lambda x: x.points)
        else:
            fit = max(results, key = lambda x: x.n_turns)
        player.reinforce_rule(fit.rule_data.rule_usage)
        print(fitness_evaluation(results, 2), player.number_rules())
        delete_last_rule(player, player.number_rules() // 100)
        for i in range(1, self.num_population):
            self.population[i] = player.copy()


@dataclass(frozen=True)
class Fitness:
    """
    DataClass to hold result data from a match
    """
    n_turns: int
    points: int
    loss: bool
    rule_data: EndResult


class GameManager:
    """
    Class to manage server and players,
    Calling __init__ creates a parallel process for the server that waits
    for the matches
    """

    def __init__(self, n_pl: int):
        """
        init method, creates a prallel process for the server
        @param n_pl: number of players
        """
        self.n_players = n_pl
        self.serv = Process(target=server.start_server, args=(self.n_players, PORT))
        self.serv.daemon = True
        self.serv.start()
        self.players = [LCSPlayer(name=f'LCS{i}') for i in range(self.n_players)]

    def action_len(self) -> int:
        """
        @return: action length
        """
        return LCSActor.get_action_length(self.n_players)

    def sensor_len(self):
        """
        @return: sensor length
        """
        return Sens.get_sensor_len(self.n_players)

    def get_fitness(self, rule_list: List[RuleSet], sensors=None) -> List[Fitness]:
        """
        Return the result of single match done between the specified rule sets
        @param rule_list: list of rulesets representing the players
        @param sensors: sensor list, if not specified uses the LCS_Sensor default one
        @return: Fitness result of the match
        """
        if sensors is None:
            sensors = Sens.package_sensors(self.n_players)
        lcs_rule_list = [LCSRules(sensors, LCSActor.get_action_length(self.n_players), rule) for rule in rule_list]
        threads = [Thread(target=player.start, args=[lcs_rule_list[i]]) for i, player in enumerate(self.players)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        results = [player.end_game_data() for player in self.players]
        data = [player.rules.end_game_data() for player in self.players]
        n_turns = max(map(lambda x: x["n_turns"], results))
        points = max(map(lambda x: x["points"], results))
        loss = max(map(lambda x: x["loss"], results))
        result = [Fitness(n_turns, points, loss, d) for d in data]
        return result

    def stop(self):
        """
        Stop the server forcibly
        @return: None
        """
        self.serv.terminate()

    def __del__(self):
        """
        Stop the server forcibly
        @return: None
        """
        self.stop()


def delete_critical_rules(rule: RuleSet, criticals) -> RuleSet:
    """
    Return a new ruleset without rules that cause criticalities
    @param rule: ruleset
    @param criticals: critical_indexes returned from fitness function
    @return: new ruleset
    """
    rule_matr = rule.pack_rules()
    return RuleSet.unpack_rules(rule_matr[~criticals, :], rule.sensor_length())


def delete_unused_rules(rule: RuleSet, matches: List[NDArray], threshold=0) -> RuleSet:
    """
    Return a new ruleset with rules that were used more then threshold
    @param rule: ruleset
    @param matches: list of rule_match returned from fitness function
    @return: new rule set
    """
    n_rules = max(matches, key=lambda x: x.shape[1]).shape[1]
    tot_usages = sum([np.append(np.sum(match, axis=0), np.zeros(n_rules - match.shape[1])) for match in matches])
    used_rules = tot_usages > threshold
    return RuleSet.unpack_rules(rule.pack_rules()[used_rules[:rule.number_rules()], :], rule.sensor_length())


def point_mutation(rule: RuleSet, p: float = 0.01) -> RuleSet:
    """
    Bit flip rule set with probability p
    @param rule: rule set
    @param p: probability of mutation
    @return: new rule set
    """
    packed_rules = rule.pack_rules()
    random_mask = np.random.choice(a=(False, True), size=packed_rules.size, p=(1 - p, p))
    random_mask = np.reshape(random_mask, packed_rules.shape)
    packed_rules ^= random_mask
    return RuleSet.unpack_rules(packed_rules, rule.sensor_length())


def delete_mutation(rule: RuleSet, p: float = 0.05):
    """
    Delete a random rule with probability p
    @param rule: The ruleSet to mutate
    @param p: the probability of deletion
    """
    num_rules = rule.number_rules()
    random_mask = np.random.choice(a=(True, False), size=num_rules, p=(1 - p, p))
    rule.dont_care = rule.dont_care[random_mask]
    rule.action = rule.action[random_mask]
    rule.match_string = rule.match_string[random_mask]


def match_mutation(rule: RuleSet, p: float = 0.05) -> RuleSet:
    """
    Like point Mutation but does not touch the action section of the rule
    @param rule: RuleSet
    @param p: Probability of mutation
    @return: new RuleSet
    """
    packed_rules = rule.pack_rules()
    random_mask = np.random.choice(a=(False, True), size=packed_rules.shape, p=(1 - p, p))
    packed_rules[:, :-6] ^= random_mask[:, :-6]
    return RuleSet.unpack_rules(packed_rules, rule.sensor_length())


def crossover_pitts_style(ruleset_a: RuleSet, ruleset_b: RuleSet, paradigm: int = 1):
    """
    Crossover between two rule sets, depending on paradigm it can be the following operation:
    1: One point crossover
    2: Two point crossover
    3: Uniform crossover
    @param ruleset_a: first rule set
    @param ruleset_b: second rule set
    @param paradigm: type of crossover
    @return: mew_ruleset, new_ruleset
    """
    p1, p2 = ruleset_a.pack_rules(), ruleset_b.pack_rules()
    s_len = ruleset_a.sensor_length()
    min_r_size = min([ruleset_a.number_rules(), ruleset_b.number_rules()])

    # One Point Crossover
    if paradigm == 0:
        cut_point = random.randint(0, min_r_size)
        p2[cut_point:, :], p1[cut_point:, :] = p1[cut_point:, :], p2[cut_point:, :]

    # Two Point Crossover
    if paradigm == 1:
        cpnt_1 = random.randint(0, min_r_size)
        cpnt_2 = random.randint(0, min_r_size)
        cpnt_1, cpnt_2 = (cpnt_1, cpnt_2) if cpnt_1 < cpnt_2 else (cpnt_2, cpnt_1)
        p1[cpnt_1:cpnt_2, :], p2[cpnt_1:cpnt_2, :] = p2[cpnt_1:cpnt_2, :], p1[cpnt_1:cpnt_2, :]

    # Uniform Crossover
    if paradigm == 2:
        rule_swap = np.random.choice([True, False], size=min_r_size, p=(.5, .5))
        cut_p1, cut_p2 = p1[:min_r_size, :], p2[:min_r_size, :]
        cut_p1[rule_swap, :], cut_p2[rule_swap, :] = cut_p2[rule_swap, :], cut_p1[rule_swap, :]
        p1[:min_r_size, :], p2[:min_r_size, :] = cut_p1, cut_p2

    return RuleSet.unpack_rules(p1, s_len), RuleSet.unpack_rules(p2, s_len)


def full_crossover(ruleset_a: RuleSet, ruleset_b: RuleSet, child_number: int):
    """
    Crossover between ruleset a and b generating specified number of children
    @param ruleset_a: first parent ruleset
    @param ruleset_b: second parent ruleset
    @param child_number: number of children
    @return: List[RuleSet]
    """
    offspring = [ruleset_a, ruleset_b]
    s_len = ruleset_a.sensor_length()
    p1, p2 = ruleset_a.pack_rules(), ruleset_b.pack_rules()
    min_r_size = min([ruleset_a.number_rules(), ruleset_b.number_rules()])
    cut_p1, cut_p2 = p1[:min_r_size, :], p2[:min_r_size, :]
    for _ in range(child_number):
        rule_swap = np.random.choice([True, False], size=min_r_size, p=(.5, .5))
        child = cut_p1.copy()
        child[rule_swap, :] = cut_p2[rule_swap, :]
        r = RuleSet.unpack_rules(child, s_len)
        r = match_mutation(r, p=0.2)
        offspring.append(r)
    return offspring


def single_crossover(sigma_male: RuleSet, child_number: int):
    """

    @param sigma_male:
    @param child_number:
    @return:
    """
    offspring = []
    for _ in range(child_number):
        child = sigma_male
        child = match_mutation(child, p=0.01)
        offspring.append(child)
    return offspring


def shuffle_rules(rule: RuleSet):
    """
    Shuffle rules randomly from a dataset
    @param rule: RuleSet
    @return: Shuffled ruleset
    """
    packed_rules = rule.pack_rules()
    np.random.shuffle(packed_rules)
    return RuleSet.unpack_rules(packed_rules, rule.sensor_length())


def delete_last_rule(rule: RuleSet, trim: int = 1):
    """
    Delete last trim rules from ruleset
    @param rule: Ruleset
    @param trim: number of rules to delete
    @return: None
    """
    rule.action = rule.action[:-trim, :]
    rule.dont_care = rule.dont_care[:-trim, :]
    rule.match_string = rule.match_string[:-trim, :]


def fitness_evaluation(match_results: List[Fitness], fit_type=0) -> float:
    """
    Evaluate fitness functions from a list of matches into a value from 0 to 1
    @param match_results: fitnesses from matches
    @param fit_type: type of fitness function
    @return: float
    """
    le = len(match_results)

    fitness = 0

    # avg(points * win) / maxpoints
    if fit_type == 0:
        fitness = sum(map(lambda x: x.points * (not x.loss), match_results)) / (le * 25)

    # (avg(points) + (sum(win)) ** 2) / (25 + n_matches**2)
    if fit_type == 1:
        fitness = (sum(map(lambda x: x.points, match_results)) / le +
                   (sum(map(lambda x: not x.loss, match_results))) ** 2) / (25 + le ** 2)

    # avg(points) / maxpoints
    if fit_type == 2:
        fitness = sum(map(lambda x: x.points, match_results)) / (25 * le)

    return fitness


def size_penality(rule: RuleSet, target_size: int):
    """
    Evaluate rule set size and return a penality if size is above target size
    @param rule: rule set
    @param target_size: size
    @return: 1 - target_size / size
    """
    if rule.number_rules() <= target_size:
        return 0

    return 1 - target_size / rule.number_rules()


def tournament_play(players: List[RuleSet], man: GameManager, repetitions: int = 1):
    """
    Play all possible combinations between players 'repetitions' times
    @param players: list of rule sets
    @param man: game manager from where to use fitness function
    @param repetitions: number of matches against the same
    @return: tuple of fitness lists for each player
    """
    result = tuple([[] for _ in range(len(players))])
    for i in range(repetitions):
        for c in combinations(range(len(players)), man.n_players):
            res = man.get_fitness([players[ix] for ix in c])
            for ix, r in zip(c, res):
                players[ix] = delete_critical_rules(players[ix], r.rule_data.critical_rules)
            [result[ix].append(r) for ix, r in zip(c, res)]
    return result


def stochastic_play(players: List[RuleSet], man: GameManager, repetitions: int = 1):
    """
    Play random matches among players
    @param players: list of rule sets
    @param man: game manager from where to use fitness function
    @param repetitions: number of matches against the same
    @return: tuple of fitness lists for each player
    """
    result = tuple([[] for _ in range(len(players))])
    n_players = man.n_players
    indx = list(range(len(players)))
    np.random.shuffle(indx)
    for _ in range(repetitions):
        for c in range(0, len(players), n_players):
            res = man.get_fitness([players[x] for x in indx[c:c+n_players]])
            for i, r in zip(indx[c: c+n_players], res):
                players[i] = delete_critical_rules(players[i], r.rule_data.critical_rules)
                result[i].append(r)
    return result


def save_LCS(lcs_rule: RuleSet, num_players: int):
    """
    Save model to folder
    @param lcs_rule: ruleset to save
    @param num_players: int
    @return: None
    """
    matr = lcs_rule.pack_rules()
    np.save(f"../hanabi/models/ruleset_{num_players}.npy", matr)
