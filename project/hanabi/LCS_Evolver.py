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

    def __init__(self, lcs_rules_list: List[RuleSet]):
        self.LCS_rules_list = lcs_rules_list

    def evolve(self, activations, fitness):
        # Activations arriva dal metodo EndGameData
        # sto metodo mi ritorna altri ruleSet
        pass


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


def delete_unused_rules(rule: RuleSet, usages: List[NDArray], threshold=0) -> RuleSet:
    """
    Return a new ruleset with rules that were used more then threshold
    @param rule: ruleset
    @param usages: list of usages returned from fitness function
    @return: new rule set
    """
    n_rules = rule.number_rules()
    tot_usages = sum([np.append(np.sum(usage, axis=0), np.zeros(n_rules - usage.shape[1])) for usage in usages])
    used_rules = tot_usages > threshold
    return RuleSet.unpack_rules(rule.pack_rules()[used_rules, :], rule.sensor_length())


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


if __name__ == '__main__':
    n_players = 3
    g = GameManager(n_players)
    population = [RuleSet.empty_rules(g.sensor_len(), g.action_len()) for _ in range(10)]

    for _ in range(10):
        print(population)
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
    g.stop()
    pass
