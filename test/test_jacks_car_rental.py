'''
Testing Jack's Car Rental

@author: Tobias Lang
'''
import unittest
import numpy as np
from jacks_car_rental import JacksCarRental


class TestJacksCarRental(unittest.TestCase):
    '''
    Testing computations for Jack's Car Rental problem.
    '''

    def setUp(self):
        self.jcr = JacksCarRental()

    def test_move_cars_positive(self):
        '''
        Check car moving for correct bounds.
        '''
        expected_new_state = [1, 12]

        state = [6, 7]
        action = 5
        actual_action, new_state = self.jcr.move_cars(state, action)
        self.assertEqual(actual_action, action)
        self.assertListEqual(new_state, expected_new_state)

    def test_move_cars_positive_too_few(self):
        '''
        Check car moving for correct bounds.
        '''
        expected_new_state = [0, 11]

        state = [4, 7]
        action = 5
        actual_action, new_state = self.jcr.move_cars(state, action)
        self.assertEqual(actual_action, 4)
        self.assertListEqual(new_state, expected_new_state)

    def test_move_cars_negative_too_few(self):
        '''
        Check car moving for correct bounds.
        '''
        expected_new_state = [14, 0]

        state = [12, 2]
        action = -5
        actual_action, new_state = self.jcr.move_cars(state, action)
        self.assertEqual(actual_action, -2)
        self.assertListEqual(new_state, expected_new_state)

    def test_bellman_update(self):
        '''
        Validate correct computation of the expected reward.
        '''
        # Precomputed reward for the given state
        exp_reward = 31.754861470593497

        state = [4, 7]
        action = 5
        expected_reward = self.jcr.bellman_optimality_update(state, action)
        self.assertAlmostEqual(expected_reward, exp_reward)

    def test_plot_policies(self):
        '''
        Test plotting final policies and state_values.
        '''

        policy = np.random.randint(low=-5, high=+5, size=(21, 21))
        state_values = np.random.rand(21, 21) * 150

        self.jcr.plot_policies(policy, state_values)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
