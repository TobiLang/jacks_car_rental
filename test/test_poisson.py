'''
Test Poisson class

@author: Tobias Lang
'''
import unittest
from poisson import Poisson


class TestPoisson(unittest.TestCase):
    '''
    Check correct lower and upper bounds and distribution calculation.
    '''

    def test_bounds(self):
        '''
        Check bounds computation
        '''
        lam = 3
        expected_lower = 0
        expected_upper = 10
        poisson = Poisson(lam)

        self.assertEqual(poisson.lower, expected_lower)
        self.assertEqual(poisson.upper, expected_upper)

    def test_probs(self):
        '''
        Check probs computation
        '''
        lam = 4
        expected_probs = [0.01831563888873418, 0.07326255555493671,
                          0.14652511110987343, 0.19536681481316456,
                          0.19536681481316456, 0.15629345185053165,
                          0.1041956345670211, 0.059540362609726345,
                          0.029770181304863173, 0.013231191691050298,
                          0.0052924766764201195, 0.0019245369732436798]

        poisson = Poisson(lam)
        for prob, expected_prob in zip(poisson.probs, expected_probs):
            self.assertAlmostEqual(prob, expected_prob)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
