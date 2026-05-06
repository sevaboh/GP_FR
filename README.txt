This repository implements a genetic programming (GP) framework for symbolic regression of equations involving fractional derivatives and integrals. 
The code in "surrogate" folder is aimed at building GP/neural network surrogate models, particularly for a fractal-fractured porous media fPDE solver ("surrogate/solver").
### `gen_prog_test.cpp`
The primary C++ implementation of the genetic programming algorithm ( https://journal.comp-sc.if.ua/test/index.php/ITCM/article/view/626 ). It evolves a population of symbolic expression trees whose nodes can represent arithmetic operators, mathematical functions (sin, exp, pow, etc.), constants, variables, fractional-order integrals and derivatives. Fitness is evaluated by comparing the GP-generated expression against target time-series data, supporting two modes: plain function fitting, first-order derivative fitting. GP parameters such as population size, mutation probability, and tree depth are loaded from a config file at runtime.

### `gen_prog_test_v2.cpp`
An extended version of `gen_prog_test.cpp` with several additions: a built-in gradient descent and genetic algorithm (GA) for fine-tuning floating-point constants within discovered GP trees; elitism and tournament selection; a solution database for reusing previously found candidates; and clustering-based substitution effect assessment. The "surrogate" workflows use this version as the main GP engine.

### `gamma.h`
A header providing an implementations for the Gamma and LogGamma functions.

### `gen_prog_run_tests`
A shell script that compiles `gen_prog_test.cpp` with OpenMP and runs it against testing datasets.

### `configt0.txt-configt5a.txt`
Testing configurations for `gen_prog_test.cpp`/`gen_prog_test_v2.cpp`.

### `rothc.csv`
Time-series output from the RothC (Rothamsted Carbon) soil organic carbon decomposition model. Each row contains time and three carbon pool values. Serves as the target dataset for GP symbolic regression in `configt1.txt`.

### `irr.csv`
Real-world hydrology and weather-related sensor time-series data. Used as the target for `configt2.txt`.

### `clC.csv`
Synthetic data representing the solution of a fractional advection–diffusion system in a given spatial point. Used with `configt3.txt`.

### `vir.csv`
A two-column time-series (time and state variable) representing growth dynamics from a viral model. Used as the target for `configt5.txt` and `configt5a.txt`.
