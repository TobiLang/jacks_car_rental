'''
Created on 18.09.2020

@author: Tobias Lang
'''

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from poisson import Poisson


class JacksCarRental:
    '''
    Jack's Car Rental Problem
    '''

    # Evaluation param
    THETA = 0.5

    # Jacks Car Rental params
    # Cars and Moving
    MAX_CARS = 20
    MAX_MOVING = 5

    # Rental and return rates
    LAMBDA_RENTAL_LOC1 = 3.0
    LAMBDA_RENTAL_LOC2 = 4.0
    LAMBDA_RETURN_LOC1 = 3.0
    LAMBDA_RETURN_LOC2 = 2.0

    # Rewards
    REWARD_RENTAL = 10
    REWARD_MOVE = -2
    REWARD_PARKING = -4

    # Discount rate
    DISCOUNT_RATE = 0.9

    def __init__(self, non_linear=False):
        '''
        Initialize all variables
        '''

        # State values, Policies
        # State values are currently unknown, +1 due to 0 cars at each location
        self.state_values = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        # We start with a no-moving policy
        self.policy = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        # Actions range from -5...+5 (moving from Loc1<->Loc2
        self.actions = np.arange(-self.MAX_MOVING, self.MAX_MOVING + 1)

        # Probabilities
        self.rental_loc1 = Poisson(self.LAMBDA_RENTAL_LOC1)
        self.rental_loc2 = Poisson(self.LAMBDA_RENTAL_LOC2)
        self.return_loc1 = Poisson(self.LAMBDA_RETURN_LOC1)
        self.return_loc2 = Poisson(self.LAMBDA_RETURN_LOC2)
        # Transitions cache
        self.total_probs = self.probabilities()

        # Activate non-linearities
        self.non_linear = non_linear

    def probabilities(self):
        '''
        Cache the probabilities
        '''
        probs = np.zeros((self.rental_loc1.upper,
                          self.rental_loc2.upper,
                          self.return_loc1.upper,
                          self.return_loc2.upper))
        for rentals_loc1 in range(self.rental_loc1.lower, self.rental_loc1.upper):
            for rentals_loc2 in range(self.rental_loc2.lower, self.rental_loc2.upper):
                for returns_loc1 in range(self.return_loc1.lower, self.return_loc1.upper):
                    for returns_loc2 in range(self.return_loc2.lower, self.return_loc2.upper):
                        prob = self.rental_loc1.probs[rentals_loc1] * \
                                self.rental_loc2.probs[rentals_loc2] * \
                                self.return_loc1.probs[returns_loc1] * \
                                self.return_loc2.probs[returns_loc2]
                        probs[rentals_loc1, rentals_loc2, returns_loc1, returns_loc2] = prob

        return probs

    def move_cars(self, state, action):
        '''
        Move cars between locations. We clip:
          * min  0: We cannot move, what we do not have
          * max 20: More than Max_Cars is not allowed.
        '''
        # How many cars can we move
        action = min(action, +state[0])
        action = max(action, -state[1])
        new_state = [int(max(min(state[0] - action, self.MAX_CARS), 0)),
                     int(max(min(state[1] + action, self.MAX_CARS), 0))]
        return action, new_state

    def bellman_optimality_update(self, state, action):
        '''
        Mutate the state values according to the Bellman Optimality Equation.
        Calculate expected reward for the given state and action
        '''
        # Calculate the number of cars available at the start of the day:
        #   current state +/- moving
        # We can only move as many cars as we have
        # State_start contains the number of cars parked at each location overnight
        # after moving cars between the locations
        actual_action, state_start = self.move_cars(state, action)
        # Cost of moving cars - we use the actual number of cars moved.
        # Reward move is negative
        expected_reward = self.REWARD_MOVE * np.absolute(actual_action)

        if self.non_linear:
            # NonLinear-extension: one car from Loc1->Loc2 is free
            if actual_action > 0:
                expected_reward -= self.REWARD_MOVE
            # NonLinear-extension: Parking more than 10 cars is expensive
            # Reward parking is negative
            parking_cost = (state_start[0] > 10) + (state_start[1] > 10) * self.REWARD_PARKING
            expected_reward += parking_cost

        # Iterate over all four Poisson bounds
        for rentals_loc1 in range(self.rental_loc1.lower, self.rental_loc1.upper):
            for rentals_loc2 in range(self.rental_loc2.lower, self.rental_loc2.upper):
                # Daily rentals at both locations
                rentals = [min(state_start[0], rentals_loc1),
                           min(state_start[1], rentals_loc2)]
                # Update reward
                reward = (rentals[0] + rentals[1]) * self.REWARD_RENTAL
                for returns_loc1 in range(self.return_loc1.lower, self.return_loc1.upper):
                    for returns_loc2 in range(self.return_loc2.lower, self.return_loc2.upper):
                        # Probability of all four events happening
                        prob = self.total_probs[rentals_loc1, rentals_loc2,
                                                returns_loc1, returns_loc2]

                        # Final state at both locations
                        new_state = [min(state_start[0] - rentals[0] + returns_loc1, self.MAX_CARS),
                                     min(state_start[1] - rentals[1] + returns_loc2, self.MAX_CARS)]

                        # Update expected reward using Bellmans Optimality Equation
                        expected_reward += prob * (
                            reward +
                            self.DISCOUNT_RATE * self.state_values[new_state[0], new_state[1]])

        return expected_reward

    def eval_policy(self, iterations=None):
        '''
        Evaluate the current policy.
        '''
        print("Evaluating Policy ...")

        while True:
            delta = 0.0

            # Loop over states
            for i, j in itertools.product(range(self.MAX_CARS + 1), range(self.MAX_CARS + 1)):
                # Save old state value
                old_state_value = self.state_values[i, j]
                # Update State Value
                new_state_value = self.bellman_optimality_update([i, j], self.policy[i, j])
                self.state_values[i, j] = new_state_value
                delta = max(delta, np.absolute((new_state_value - old_state_value)))

            # Evaluate Theta condition
            if delta < self.THETA:
                break

            # Are iterations given?
            if iterations:
                iterations -= 1
                if iterations <= 0:
                    break

        return delta

    def improve_policy(self):
        '''
        Greedy policy improvement using the current value states and picking the action with
        the highest value return according to the Bellmann Optimality Equation.
        '''
        print("Improving Policy ...")

        policy_stable = True
        for i, j in itertools.product(range(self.MAX_CARS + 1), range(self.MAX_CARS + 1)):
            old_action = self.policy[i, j]
            # Action with best expected return
            max_action = None
            max_action_return = -np.inf
            # How many cars can be moved for the given state
            # Negative numbers mean moving cars to location 1
            move_to_loc2 = min(i, 5)
            move_to_loc1 = -min(j, 5)
            # Ignore all other actions - as they are not possible.
            # We can only move cars that actually exist
            for action in range(move_to_loc1, move_to_loc2 + 1):
                expected_return = self.bellman_optimality_update([i, j], action)
                if expected_return > max_action_return:
                    max_action = action
                    max_action_return = expected_return
            self.policy[i, j] = max_action
            if self.policy[i, j] != old_action:
                policy_stable = False

        return policy_stable

    def plot_policies(self, policy, state_values):
        '''
        Plot the given policy and state_value function
        '''

        fig = plt.figure(figsize=(12, 12))
        # Plot Policy
        ax_policy = fig.add_subplot(121, aspect='equal')
        im_policy = ax_policy.imshow(policy, cmap=plt.cm.get_cmap('magma'))
        plt.ylabel('Cars at Location 1')
        plt.xlabel('Cars at Location 2')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax_policy)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_policy, cax=cax)

        # Plot State-Values
        ax_state = fig.add_subplot(122, aspect='equal')
        im_state = ax_state.imshow(state_values, cmap=plt.cm.get_cmap('magma'))
        plt.ylabel('Cars at Location 1')
        plt.xlabel('Cars at Location 2')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax_state)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_state, cax=cax)

        if self.non_linear:
            plt.savefig("NonLinear_Jack.png")
        else:
            plt.savefig("Plain_Jack.png")

    def run(self):
        '''
        Run Policy Iteration algorithm on Jack's Car Rental problem.
        '''

        while True:
            # Eval Policy
            self.eval_policy()
            # Improve Policy
            stable = self.improve_policy()

            # Policy stable?
            if stable:
                break

        self.plot_policies(self.policy, self.state_values)


if __name__ == '__main__':
    JCR = JacksCarRental(non_linear=False)
    JCR.run()
