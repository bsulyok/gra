# gready routing annealing


Run with default parameters:

```
import gra
import networkx as nx

G = nx.barabasi_albert_graph(100, 2)

coords = gra.embed(
    G=G,
    maxiter=100,
    target_function=gra.score_functions.GreedyRoutingSuccessRate,
    state_generator=gra.state_generators.RandomSampling,
    temperature=lambda step: 1/(1+step),
    coords=gra.embedding.random(G),
    exclude_neighbours=False,
    distance_function=gra.geometry.native_disk_distance,
)
```

# Parameters

### maxiter
number of iterations of the simulated annealing process

### target_function
embedding quality function to optimize against, currently two possible functions are implemented:
- gra.score_functions.GreedyRoutingSuccessRate
- gra.score_functions.GeometricalCongruence

### state_generator

a function that generates new states with arbitrary change, currently the following are implemented:
- gra.state_generators.RandomSampling
- gra.state_generators.DegreeDependentSampling
- gra.state_generators.SourceDependentSampling
- gra.state_generators.TargetDependentSampling
- gra.state_generators.FixedHubSampling

the latter two are based on the ratio of paths successfully reaching the target from the source,
these are not applicable when optimizing for GeometricalCongruence

### temperature

two possible ways to control the temperature are implemented:
- iterable: a array-like of length maxiter which specifies the value of temperature at each time step
- callable: a function that takes as input the number of steps completed and returns the temperature

### coords

initial coordinates, must be a python dictionary with the vertex IDs as keys and 2-dimensional cartesian coordinates as values

### exclude_neighbours

a binary value specifying whether to exclude neighbour vertices as source-target pairs when calculating the embedding scores

### distance_function

the distance function used to calculate vertex distances of vertex pairs during the simulation process,
we studies only the native disk distance case nevertheless we implemented the following:
- gra.geometry.native_disk_distance
- gra.geometry.euclidean_distance
- gra.geometry.cosine_similarity