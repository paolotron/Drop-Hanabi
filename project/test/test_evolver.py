import unittest
import LCS_Evolver as Ev
import LCS_Rules as Rule
import LCS_Sensor as Sens
import numpy as np


class EvolverTest(unittest.TestCase):
    def test_bootstrap(self):
        boot = Ev.bootstrap_rules(4)
        ruleset = Rule.RuleSet.unpack_rules(boot, Sens.get_sensor_len(4))
        self.assertEqual(np.unique(ruleset.action, axis=1).shape, ruleset.action.shape)

    def test_bootsrapped_player(self):
        n_players = 5
        man = Ev.GameManager(n_players)
        boot = Ev.bootstrap_rules(n_players)
        p = [Rule.RuleSet.unpack_rules(boot, man.sensor_len()) for _ in range(n_players)]
        fit = man.get_fitness(p)
        print(f"{fit[0].points}, {fit[0].loss}")
        man.stop()


if __name__ == '__main__':
    unittest.main()
