'''
The cars rented and returned at Jack's follow a Poisson distribution.

Moreover, all rentals and returns are independent of each other. Thus,
the total probability of given by p(s',a|s,a) for rentals and returns at
both locations is given by their product.

To avoid looping over extremly small probabilities we use lower and upper
bounds for the distributions.


@author: Tobias Lang
'''

from scipy.stats import poisson


class Poisson(object):
    '''
    Implements Poisson distribution for Jack's Car Rental.
    Using a lower and upper bound to avoid unnecessary looping.
    '''

    CONFIDENCE_INTERVALL = 0.999

    def __init__(self, lam):
        '''
        Setup lamba for the poisson distribution.
        Calculate lower and upper bounds.
        '''
        self.lam = lam

        # Calculate Lower and Upper Bounds for the given confidence interall
        self.lower, self.upper = poisson.interval(self.CONFIDENCE_INTERVALL, self.lam)
        self.lower = int(self.lower)
        self.upper = int(self.upper)

        # Caculate probabilities within the given bounds
        self.probs = [poisson.pmf(k, self.lam) for k in range(self.lower, self.upper)]
