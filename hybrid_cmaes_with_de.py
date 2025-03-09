import numpy as np
import torch
from deap import base, creator, tools, cma
import os

class HybridCMAESWithDE:
    def __init__(self, min_weight, max_weight, last_layer_shape, validation_class, de_rate=0.2):
        """
        Initialize the hybrid optimizer.

        Args:
            min_weight: Minimum weight value.
            max_weight: Maximum weight value.
            last_layer_shape: Shape of the last layer weights.
            validation_class: ModelValidator instance for evaluating fitness.
            de_rate: Fraction of the population for DE refinement.
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.last_layer_shape = last_layer_shape
        self.validation_class = validation_class
        self.de_rate = de_rate  # Fraction of individuals to refine using DE

        # DEAP Setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, self.min_weight, self.max_weight)
        self.toolbox.register(
            "individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=np.prod(last_layer_shape)
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # DEAP Operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selRoulette)
        self.toolbox.register("evaluate", self.fitness_function)

        # Initialize CMA-ES
        self.cma_strategy = cma.Strategy(
            centroid=torch.zeros(np.prod(last_layer_shape), dtype=torch.float32, device="cuda").tolist(),
            sigma=0.5,
            lambda_=30  # Smaller population
        )
        self.toolbox.register("generate", self.cma_strategy.generate, creator.Individual)
        self.toolbox.register("update", self.cma_strategy.update)

    def fitness_function(self, individual):
        """
        Evaluate fitness by computing model accuracy.

        Args:
            individual: Individual weight values.

        Returns:
            Fitness value (accuracy).
        """
        individual_tensor = torch.FloatTensor(individual).reshape(self.last_layer_shape).to(self.validation_class.device)
        accuracy = self.validation_class.validate_model_with_weights(individual_tensor, 5, 10)
        return accuracy,

    def _evaluate_population_with_cuda_streams(self, population, generation , total_generations):
        """Evaluate fitness of the population using CUDA streams."""
        streams = [torch.cuda.Stream(device=self.validation_class.device) for _ in range(len(population))]
        fitness_results = [None] * len(population)

        # Start evaluation in parallel streams
        for i, (ind, stream) in enumerate(zip(population, streams)):
            with torch.cuda.stream(stream):
                weights = torch.FloatTensor(ind).reshape(self.last_layer_shape).to(self.validation_class.device)
                fitness_results[i] = self.validation_class.validate_model_with_weights(weights, generation, total_generations)

        # Synchronize all streams
        torch.cuda.synchronize(self.validation_class.device)
        return fitness_results

    def differential_evolution(self, population, fitness_dict, generation, total_generations):
        """
        Perform Differential Evolution (DE) on a subset of the population.

        Args:
            population: List of individuals.

        Returns:
            Updated population after applying DE.
        """
        F = 0.8  # DE scaling factor
        CR = 0.80  # Crossover probability

        # Ensure all individuals have valid fitness values
        print("Finding Fitness for Solutions in Existing Population")
        for ind in population:
            if not ind.fitness.valid:
                print(ind.fitness.values[0])
                fitness = self.toolbox.evaluate(ind)
                ind.fitness.values = fitness

        print("Finding Fitness for Trial Solutions")
        refined_population = []
        trial_vectors = []
        target_indices = []

        # Generate all trial vectors
        for target_idx in range(len(population)):
            target = population[target_idx]
            idxs = [i for i in range(len(population)) if i != target_idx]
            if len(idxs) < 3:
                print("Skipping DE for this individual due to insufficient population.")
                refined_population.append(target)
                continue

            # Select three random, distinct individuals
            a, b, c = np.random.choice(idxs, 3, replace=False)
            donor = [
                population[a][i] + F * (population[b][i] - population[c][i])
                for i in range(len(target))
            ]
            trial = [
                donor[i] if np.random.rand() < CR else target[i]
                for i in range(len(target))
            ]

            # Store the trial vector and the target index
            trial_vectors.append(trial)
            target_indices.append(target_idx)

        # Evaluate all trial vectors in parallel
        trial_fitnesses = self._evaluate_population_with_cuda_streams(
            [creator.Individual(trial) for trial in trial_vectors],
            generation, total_generations
        )

        # Perform selection based on fitness
        for trial_vector, trial_fitness, target_idx in zip(trial_vectors, trial_fitnesses, target_indices):
            target = population[target_idx]
            if trial_fitness > target.fitness.values[0]:
                new_ind = creator.Individual(trial_vector)
                new_ind.fitness.values = (trial_fitness,)
                refined_population.append(new_ind)
            else:
                refined_population.append(target)

        return refined_population
    def run(self, population_size=100, num_generations=50, stop_threshold=95, save_interval=10, checkpoint_file="hybrid_checkpoint.pt"):
        """
        Run the hybrid CMA-ES + DE algorithm with periodic checkpointing.

        Args:
            population_size: Size of the population.
            num_generations: Number of generations to run.
            stop_threshold: Accuracy threshold to stop the optimization.
            save_interval: Number of generations between checkpoint saves.
            checkpoint_file: Path to save the checkpoint file.

        Returns:
            The best individual found during the optimization.
        """
        start_generation = 1
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.cma_strategy = checkpoint["cma_strategy"]
            start_generation = checkpoint["generation"] + 1
            print(f"Resuming from generation {start_generation}.")

        # Initialize refined population
        refined_population = None

        for generation in range(start_generation, num_generations + 1):
            print(f"Generation {generation}")

            # If we have a refined population, use it; otherwise, generate a new one
            if refined_population:
                cma_population = refined_population
            else:
                cma_population = self.toolbox.generate()

            fitness_dict = {}  # Key: individual ID, Value: fitness


            fitnesses = self._evaluate_population_with_cuda_streams(cma_population, generation, num_generations)
            for ind, fitness in zip(cma_population, fitnesses):
                ind.fitness.values = (fitness,)
                fitness_dict[id(ind)] = ind.fitness.values 

            # Update CMA-ES with evaluated fitnesses
            self.cma_strategy.update(cma_population)

            # Select the best CMA-ES individual
            best_cma_ind = max(cma_population, key=lambda ind: ind.fitness.values[0])
            print(f"Best CMA-ES accuracy: {best_cma_ind.fitness.values[0]:.2f}%")

            # Check stopping criterion
            if best_cma_ind.fitness.values[0] >= stop_threshold:
                print("Stopping criterion reached.")
                return best_cma_ind

            # Apply DE to refine a subset of the population
            de_population = [creator.Individual(ind) for ind in cma_population]
            # Copy fitness from original individuals
            for new_ind, original_ind in zip(de_population, cma_population):
                new_ind.fitness.values = original_ind.fitness.values
            
            refined_population = self.differential_evolution(de_population, fitness_dict, generation, num_generations)

            # Determine the number of individuals to reuse and generate
            num_to_reuse = population_size // 2  # Top 50% from refined population
            num_to_generate = population_size - num_to_reuse  # Remaining 50% to generate

            # Reuse top individuals from the refined population
            top_refined_population = sorted(refined_population, key=lambda ind: ind.fitness.values[0], reverse=True)[:num_to_reuse]

            # Generate the remaining individuals
            new_cma_population = self.toolbox.generate()[:num_to_generate]

            print(f"length of new cma population = {len(new_cma_population)}")

            fitnesses_new_pop = self._evaluate_population_with_cuda_streams(new_cma_population, generation, num_generations)
            # Calculating Fitness for newly generated Individuals
            for ind, fitness in zip(new_cma_population, fitnesses_new_pop):
                ind.fitness.values = (fitness,)
                fitness_dict[id(ind)] = ind.fitness.values

            # Combine reused and newly generated populations
            combined_population = top_refined_population + new_cma_population

            # Sort combined population by fitness (descending)
            combined_population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)

            # Retain the top population_size individuals for the next generation
            refined_population = combined_population[:30]

            # Save checkpoint at specified intervals
            if generation % save_interval == 0:
                checkpoint_data = {
                    "cma_strategy": self.cma_strategy,
                    "generation": generation,
                }
                torch.save(checkpoint_data, checkpoint_file)
                print(f"Checkpoint saved at generation {generation}.")

        return best_cma_ind