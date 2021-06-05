%pylab inline

from numba import jit
import random

def monte_carlo_pi(nsamples, a):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if(x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

%time monte_carlo_pi(10000, 10000)

b = jit(nopython=True)(monte_carlo_pi)
%time b(10000, 10000)
