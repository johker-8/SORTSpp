import population
import numpy as np

pop0 = population.Population(
        name='test objects',
    )

pop = population.Population(
        name='two objects',
        extra_columns = ['m', 'color'],
        dtypes = ['Float64', 'U20'],
        space_object_uses = [True, False],
    )

print(pop)

pop0.allocate(2)
pop.allocate(2)

print(pop)

print(pop[1])
print(pop['m'])
print(pop[1,5])
print(pop[1,:])
print(pop[:,3:])

pop[1,5] = 1
pop[1,9] = 'WAT'
pop0[0] = np.random.randn(len(pop.header))
pop['m'] = np.random.randn(len(pop))
pop[:,3] = np.ones((len(pop),1))*3
pop[:,5:8] = np.ones((len(pop),3))*4
print(pop)