# Vehicle Routing Problem

## Introduction
### Problem statement 
The objective is to minimize the allocation of vehicles and their capacities to customer requests in order to minimize various costs, such as travel time, total distance travelled, or the number of vehicles used,while satisfying all the constraints

### DVRP
 The Vehicle Routing Problem is a puzzle where we need to optimize and find the best routes 
for vehicle in traffic, to get things done quickly and efficiently. 
The traditional vehicle routing problem assumes that all routes and travel times are known in advance, leading to static solutions.However, in a dynamic environment where traffic conditions change frequently, a new approach is required.This involves adjusting vehicle routes dynamically to ensure optimal performance under constantly changing conditions.
 Dynamic Vehicle Routing is like regular vehicle routing, but it's way more flexible because it 
gets adjusted to unpredictable events in real-time. This means the routes are constantly being 
updated to handle the new information time to time.

## Algorithm
### Genetic algorithm
A Genetic Algorithm (GA) is an optimization technique inspired by the process of natural selection. It is used to find high-quality solutions to complex problems by evolving a population of candidate solutions over time.Key features of genetic algorithm are

**Population**:The population is a set of possible solutions, where each solution represents a complete assignment of routes to vehicles. A diverse population helps explore a wide search space and increases the chances of finding optimal or near-optimal solutions.

**Chromosome**:A chromosome encodes one potential solution, typically represented as a sequence indicating the order in which customers are visited. In DVRP, it must also respect constraints like vehicle capacity and delivery deadlines.

**Fitness Function**:The fitness function evaluates the quality of each solution by calculating metrics such as total travel time, route efficiency, or fuel cost. In this project, it also factors in dynamic traffic conditions to prefer less congested paths.

**Selection**:Selection chooses the fittest individuals from the current population to serve as parents for the next generation. This ensures that better-performing solutions are more likely to propagate their traits.

**Crossover (Recombination)**:Crossover combines parts of two parent chromosomes to produce one or more offspring. This operation encourages the sharing of beneficial traits, such as efficient sub-routes or balanced workloads.

**Mutation**:Mutation introduces small random changes to a chromosome, such as swapping two delivery points. It prevents premature convergence by maintaining genetic diversity and allowing the exploration of new areas in the solution space.




