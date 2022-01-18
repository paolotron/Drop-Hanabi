from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Process
from threading import Thread
from typing import List

import numpy as np
from numpy.typing import NDArray

import LCS_Sensor as Sens
import server_custom as server
from LCS_Actor import LCSActor
from LCS_Rules import LCSRules, EndResult
from LCS_Rules import RuleSet
from constants import *
from player_LCS import LCSPlayer


class Evolver:

    def __init__(self, num_players):
        self.num_players = num_players
        self.num_population = 10
        self.gameManager = GameManager(num_players)
        self.rules_set = [RuleSet.empty_rules(self.gameManager.sensor_len(), self.gameManager.action_len())
                          for _ in range(self.num_population)]

    def evolve(self):
        results = tournament_play(self.rules_set, self.gameManager, 1)

        points = np.zeros(len(self.rules_set))
        for i, player in enumerate(results):
            tot = 0
            for fit in player:
                fit: Fitness
                tot += fit.points
            points[i] = tot
        ind_p1, ind_p2 = points.argpartition(-2)[-2:]
        print(points[ind_p1] / len(results[0]), points[ind_p2] / len(results[0]), self.rules_set[0].number_rules())
        self.rules_set = full_crossover(self.rules_set[ind_p1], self.rules_set[ind_p2], self.num_population - 2)

        # sto metodo mi ritorna altri ruleSet
        return self.rules_set


@dataclass(frozen=True)
class Fitness:
    n_turns: int
    points: int
    loss: bool
    rule_data: EndResult


class GameManager:

    def __init__(self, n_pl):
        self.n_players = n_pl
        self.serv = Process(target=server.start_server, args=(self.n_players, PORT))
        self.serv.daemon = True
        self.serv.start()
        self.players = [LCSPlayer(name=f'LCS{i}') for i in range(self.n_players)]

    @staticmethod
    def action_len():
        return LCSActor.get_action_length()

    def sensor_len(self):
        return Sens.get_sensor_len(self.n_players)

    def get_fitness(self, rule_list: List[RuleSet], sensors=None) -> List[Fitness]:
        if sensors is None:
            sensors = Sens.package_sensors(self.n_players)
        lcs_rule_list = [LCSRules(sensors, LCSActor.get_action_length(), rule) for rule in rule_list]
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
        self.serv.terminate()

    def __del__(self):
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


def point_mutation(rule: RuleSet, p: float = 0.01, copy=False) -> RuleSet:
    """
    Bit flip rule set with probability p
    @param rule: rule set
    @param p: probability of mutation
    @param copy: if True does not modify the parent creating a copy
    @return: new rule set
    """
    packed_rules = rule.pack_rules()
    random_mask = np.random.choice(a=(False, True), size=packed_rules.size, p=(1 - p, p))
    random_mask = np.reshape(random_mask, packed_rules.shape)
    if copy:
        new_packed_rules = np.copy(packed_rules)
        new_packed_rules ^= random_mask
        return RuleSet.unpack_rules(new_packed_rules, rule.sensor_length())
    else:
        packed_rules ^= random_mask
        return RuleSet.unpack_rules(packed_rules, rule.sensor_length())


def delete_mutation(rule: RuleSet, p: float = 0.05):
    num_rules = rule.number_rules()
    random_mask = np.random.choice(a=(True, False), size=num_rules, p=(1 - p, p))
    rule.dont_care = rule.dont_care[random_mask]
    rule.action = rule.action[random_mask]
    rule.match_string = rule.match_string[random_mask]


def match_mutation(rule: RuleSet, p: float = 0.05) -> RuleSet:
    packed_rules = rule.pack_rules()
    random_mask = np.random.choice(a=(False, True), size=packed_rules.shape, p=(1 - p, p))
    packed_rules[:, 6:] ^= random_mask[:, 6:]
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
    offspring = [ruleset_a, ruleset_b]
    s_len = ruleset_a.sensor_length()
    p1, p2 = ruleset_a.pack_rules(), ruleset_b.pack_rules()
    min_r_size = min([ruleset_a.number_rules(), ruleset_b.number_rules()])
    cut_p1, cut_p2 = p1[:min_r_size, :], p2[:min_r_size, :]
    for _ in range(child_number):
        rule_swap = np.random.choice([True, False], size = min_r_size, p = (.5, .5))
        child = cut_p1.copy()
        child[rule_swap, :] = cut_p2[rule_swap, :]
        r = RuleSet.unpack_rules(child, s_len)
        # delete_mutation(r)
        r = match_mutation(r, p=0.2)
        offspring.append(r)
    return offspring


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


def dummy_play(n_players):
    # TODO test function remove it from final version
    man = GameManager(n_players)
    for i in range(100):
        r1 = RuleSet.random_rule_set(man.sensor_len(), man.action_len(), 100)
        r2 = RuleSet.random_rule_set(man.sensor_len(), man.action_len(), 100)
        j = man.get_fitness([r1, r2])
        j2 = man.get_fitness([r1, r2])
        if i % 10 == 0:
            print(i)


def bootstrap_rules(n_players: int) -> NDArray:
    n_cards = 4 if n_players <= 3 else 5
    sens = Sens.package_sensors(n_players)
    len_rule = Sens.get_sensor_len(n_players)
    n_rules = n_cards*2
    rule_set = np.zeros((n_rules, len_rule*2 + LCSActor.get_action_length()), dtype=bool)
    for i in range(n_cards):
        idx = Sens.get_debug_string(n_players, Sens.SurePlaySensor, i)
        idx2 = Sens.get_debug_string(n_players, Sens.RiskyPlaySensor, i)
        rule_set[i, idx] = True
        rule_set[i, len_rule: len_rule*2] = True
        rule_set[i, idx + len_rule] = False
        rule_set[i+n_cards, idx2] = True
        rule_set[i+n_cards, len_rule: len_rule * 2] = True
        rule_set[i+n_cards, idx2 + len_rule] = False
        act = list(map(lambda x: bool(int(x)), str(bin(i))[2:][::-1]))
        for j, a in enumerate(act):
            rule_set[i, len_rule*2 + j] = a
            rule_set[i+n_cards, len_rule*2 + j] = a

    idx_hint = Sens.get_debug_string(n_players, Sens.NoHintLeftSensor, 0)
    for i in range(n_players):
        for j in range(5):
            idx = Sens.get_debug_string(n_players, Sens.HintNumberToPlaySensor, j)
            idx2 = Sens.get_debug_string(n_players, Sens.HintColorToPlaySensor, j)
            act = list(map(lambda x: bool(int(x)), str(bin(n_cards*2 + i*10 + j))[2:][::-1]))
            act2 = list(map(lambda x: bool(int(x)), str(bin(n_cards*2 + i*10 + j + 5))[2:][::-1]))
            rule_set[n_cards*2 + i*10 + j, idx] = True
            rule_set[n_cards*2 + i*10 + j, len_rule: len_rule*2] = True
            rule_set[n_cards*2 + i*10 + j, idx + len_rule] = False
            rule_set[n_cards*2 + i*10 + j, idx_hint + len_rule] = False
            rule_set[n_cards*2 + i*10 + j + 5, idx2] = True
            rule_set[n_cards*2 + i*10 + j + 5, len_rule: len_rule * 2] = True
            rule_set[n_cards*2 + i*10 + j + 5, idx2 + len_rule] = False
            rule_set[n_cards*2 + i*10 + j + 5, idx_hint + len_rule] = False
            for k, a in enumerate(act):
                rule_set[n_cards*2 + i*10 + j, len_rule*2 + k] = a
            for k, a in enumerate(act2):
                rule_set[n_cards*2 + i*10 + j + 5, len_rule*2 + k] = a

    for i in range(n_players):
        for j in range(10):
            idx = Sens.get_debug_string(n_players, Sens.HintToDiscardSensor, j)
            act = list(map(lambda x: bool(int(x)), str(bin(n_cards*2 + i*10 + j))[2:][::-1]))
            rule_set[n_cards*2 + n_players*10 + i*10 + j, idx] = True
            rule_set[n_cards*2 + n_players*10 + i*10 + j, len_rule: len_rule*2] = True
            rule_set[n_cards*2 + n_players*10 + i*10 + j, idx + len_rule] = False
            rule_set[n_cards*2 + n_players*10 + i*10 + j, idx_hint + len_rule] = False
            for k, a in enumerate(act):
                rule_set[n_cards*2 + n_players*10 + i*10 + j, len_rule*2 + k] = a

    for i in range(n_cards):
        idx = Sens.get_debug_string(n_players, Sens.DiscardKnowSensor, i)
        idx2 = Sens.get_debug_string(n_players, Sens.DiscardUnknownSensor, i)
        rule_set[n_cards*2 + n_players*10*2 + i, idx] = True
        rule_set[n_cards*2 + n_players*10*2 + i, len_rule: len_rule*2] = True
        rule_set[n_cards*2 + n_players*10*2 + i, idx + len_rule] = False
        rule_set[n_cards*2 + n_players*10*2 + i + n_cards, idx2] = True
        rule_set[n_cards*2 + n_players*10*2 + i + n_cards, len_rule: len_rule * 2] = True
        rule_set[n_cards*2 + n_players*10*2 + i + n_cards, idx2 + len_rule] = False
        act = list(map(lambda x: bool(int(x)), str(bin(i+n_cards))[2:][::-1]))
        for j, a in enumerate(act):
            rule_set[n_cards*2 + n_players*10*2 + i, len_rule*2 + j] = a
            rule_set[n_cards*2 + n_players*10*2 + i + n_cards, len_rule*2 + j] = a
    return rule_set


if __name__ == '__main__':
    n_players = 3
    g = GameManager(n_players)
    population = [RuleSet.empty_rules(g.sensor_len(), g.action_len()) for _ in range(10)]

    for _ in range(10):
        fit = tournament_play(population, g, 1)
        result = []
        for i, f in enumerate(fit):
            result.append((i, sum(elem.points for elem in f)))
        result.sort(key=lambda x: x[1])
        new_pop = []
        for e in result[5:]:
            new_pop.append(population[e[0]])
        for _ in range(2):
            ret = crossover_pitts_style(new_pop[3], new_pop[4], 2)
            new_pop.append(ret[0])
            new_pop.append(ret[1])
        new_pop.append(point_mutation(new_pop[4], 0.05))
        population = new_pop.copy()

        print(fit)
    g.stop()
