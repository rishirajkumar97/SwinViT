import numpy as np
import torch
from deap import base, creator, tools

class GeneticAlgorithmEvolution:
    def __init__(self, min_weight, max_weight, last_layer_shape, validation_class):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.last_layer_shape = last_layer_shape
        self.validation_class = validation_class

        # DEAP Setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, self.min_weight, self.max_weight)
        self.toolbox.register(
            "individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=np.prod(last_layer_shape)
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Use Two-Point Crossover
        self.toolbox.register("mate", tools.cxTwoPoint)

        # Use Gaussian Mutation with adaptive probability (set dynamically in `run`)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)

        # Use Roulette Selection
        self.toolbox.register("select", tools.selRoulette)

    def fitness_function(self, individual):
        """Evaluate fitness by setting weights and computing accuracy."""
        individual_tensor = torch.FloatTensor(individual).reshape(self.last_layer_shape)
        accuracy = self.validation_class.validate_model_with_weights(individual_tensor)
        return accuracy,

    def run(self, population_size=100, num_generations=50, stop_threshold=93):
        """Run the genetic algorithm."""
        self.toolbox.register("evaluate", self.fitness_function)

        population = self.toolbox.population(n=population_size)

        for generation in range(num_generations):
            print(f"Generation {generation + 1}")

            # Evaluate fitness
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Check for stopping criterion
            best_ind = tools.selBest(population, k=1)[0]
            best_fitness = best_ind.fitness.values[0]
            print(f"Best accuracy in Generation {generation + 1}: {best_fitness:.2f}%")

            if best_fitness >= stop_threshold:
                print("Stopping criterion reached.")
                return best_ind

            # Evolve population
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply Two-Point Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Adaptive Mutation Probability
            mutation_prob = 1 - (generation / num_generations)  # Decrease mutation probability over generations
            for mutant in offspring:
                if np.random.rand() < mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Replace population with offspring
            population[:] = offspring

        # Return the best individual from the final generation
        return tools.selBest(population, k=1)[0]