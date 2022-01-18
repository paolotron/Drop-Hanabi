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

    def test_bootstrpped_player(self):
        man = Ev.GameManager(4)
        boot = Ev.bootstrap_rules(4)
        p1, p2, p3, p4 = [Rule.RuleSet.unpack_rules(boot, man.sensor_len()) for _ in range(4)]
        fit = man.get_fitness([p1, p2, p3, p4])
        pass

if __name__ == '__main__':
    unittest.main()
