# Jack's Car Rental

This is a classical Dynamic Programming problem from the book [Reinforcement Learning, by Sutton and Barto](https://mitpress.mit.edu/books/reinforcement-learning-second-edition).

This is my solution to the problem - as there are many on the web.

I used this to get comfortable with te terminology (StateValues, Policies, Actions, States, ...) and to have a hands-on project on Dynamic Programming.
Moreover, I learned some of the tricks on how to tackle such problems. For example, using boundaries to avoid computing unimportant longtail probabilities. And this drastically speeds up computation. However, I did not vectorize the solution.

The script is pretty straightforward, and I added the non-linear extensions via a switch.

## Original Problem:

This is just a copy of the textbook tasks (to spare the lookup):

> Example 4.2: Jack’s Car Rental Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n is $\frac{\lambda^n}{n!] \exp^{-}lambda}$ , where is the expected number. Suppose is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a
maximum of five cars can be moved from one location to the other in one night. We take
e discount rate to be = 0.9 and formulate this as a continuing finite MDP, where
the time steps are days, the state is the number of cars at each location at the end of
the day, and the actions are the net numbers of cars moved between the two locations
overnight

## Nonlinear Extension:

And this is the non-linear extension to the original problem. The script has an initialization parameter to switch between both versions:

> Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
rental problem with the following changes. One of Jack’s employees at the first location
rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs $2, as do all cars
moved in the other direction. In addition, Jack has limited parking space at each location.
If more than 10 cars are kept overnight at a location (after any moving of cars), then an
additional cost of $4 must be incurred to use a second parking lot (independent of how
many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
occur in real problems and cannot easily be handled by optimization methods other than
dynamic programming. To check your program, first replicate the results given for the
original problem.