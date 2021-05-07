# FFEAT

Framework For Evolutionary Algorithms in Torch

--------------

This library implements various evolutionary algorithms, specifically
- Genetic Algorithms in `ffeat.genetic` module.
- Real-Coded Evolutionary Algorithms in `ffeat.strategies` module.
- Evolution Strategies in `ffeat.strategies` module.
- Particle Swarm Optimisation in `ffeat.pso` module.

The algorithms are fully vectorized and can run on GPU.

Each module consists of `selection`, `crossover`, and `mutation` submodule implementing relevant operators
(with the exception of PSO algorithm).
The operators may be arbitrarily combined.

See examples for more information on how to use the library.

This library was developed as part of my master thesis: https://github.com/PatrikValkovic/MasterThesis.
You can find more information about the implementation there.

--------------

Author: Patrik Valkovič

License: MIT

