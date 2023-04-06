# IBaCoP2 Unified Planning Integrator
This is an integrator for the portfolio selector IBaCoP2.

The Instance-Based Configurator of Portfolio-2 (IBaCoP2) is a portfolio selector  
## Installation
Clone this repository and put the directory "up_ibacop" in your python library. 

IBaCoP2 and therefore this integrator requires:
+ To be run on Linux OS (Version tested: Ubuntu 22.04.1 LTS)
+ Python >=3.7

It is heavily advised to also install Python 2.7 as without it the ordered list created won't be as accurate.

## Usage
```python
from unified_planning.shortcuts import *

x = Fluent("x")

a = InstantaneousAction("a")
a.add_precondition(Not(x))
a.add_effect(x, True)

problem = Problem("basic")
problem.add_fluent(x)
problem.add_action(a)
problem.set_initial_value(x, False)
problem.add_goal(x)

max_planners = 2

with PortfolioSelector(name="ibacop") as portfolioSelector:
    planners, parameters = portfolioSelector.get_best_oneshot_planners(
        problem, max_planners
    )

for single_planner, single_param in zip(planners, parameters):
    with OneshotPlanner(name=single_planner, params=single_param) as planner:
        result = planner.solve(problem)
        if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:
            print(f"{planner.name} found this plan: {result.plan}")
        else:
            print("No plan found.")

```
          
## Default Configuration
Currently the configurator makes use of a predictive model containing the following planners: *ENSHP*, *LPG*, *Fast-Downward*, *Tamer*. Those planners are used with their default configurations.

## Operative modes of UP currently used
PortfolioSelector
## Acknowledgements
<img src="https://www.aiplan4eu-project.eu/wp-content/uploads/2021/07/euflag.png" width="60" height="40">

This library is being developed for the AIPlan4EU H2020 project (https://aiplan4eu-project.eu) that is funded by the European Commission under grant agreement number 101016442.

It integrates the IBaCoP2 configurator, which is developed by Isabel Cenamor and Tomas de la Rosa and Fernando Fernandez. More information can be found at *The IBaCoP Planning System: Instance-Based Configured Portfolios.* PhD thesis, Universidad Carlos III de Madrid, 2016. 


