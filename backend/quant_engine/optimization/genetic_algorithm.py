# backend/quant_engine/optimization/genetic_algorithm.py
"""
Genetic Algorithm for Portfolio Optimization

Implements a genetic algorithm to find optimal portfolio weights by evolving
a population of candidate solutions. Useful for non-convex optimization problems
and incorporating complex constraints that are difficult to handle with traditional methods.

Features:
- Multi-objective optimization (return vs risk)
- Custom fitness functions
- Elitism and tournament selection
- Adaptive mutation rates
- Constraint handling
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger


class GeneticPortfolioOptimizer:
    """
    Genetic Algorithm for portfolio weight optimization

    The algorithm evolves a population of portfolios over multiple generations,
    selecting the fittest individuals for reproduction and applying mutations
    to explore the solution space.

    Chromosome representation: [w1, w2, ..., wn] where wi = weight of asset i
    """

    def __init__(
        self,
        n_assets: int,
        population_size: int = 100,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,
        tournament_size: int = 5,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        random_seed: int | None = None,
    ):
        """
        Initialize genetic algorithm optimizer

        Args:
            n_assets: Number of assets in portfolio
            population_size: Size of population (must be even)
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation (0 to 1)
            crossover_rate: Probability of crossover (0 to 1)
            elitism_rate: Fraction of top performers to preserve (0 to 1)
            tournament_size: Number of individuals in tournament selection
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            random_seed: Random seed for reproducibility
        """
        self.n_assets = n_assets
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.n_elites = int(self.population_size * elitism_rate)

        # Evolution tracking
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.diversity_history: list[float] = []

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(
            f"Genetic Algorithm initialized: pop={population_size}, "
            f"gen={n_generations}, assets={n_assets}"
        )

    def initialize_population(self) -> np.ndarray:
        """
        Create initial population of random portfolios

        Each individual is a valid portfolio (weights sum to 1, within bounds)

        Returns:
            Population matrix of shape (population_size, n_assets)
        """
        population = np.random.random((self.population_size, self.n_assets))

        # Clip to bounds
        population = np.clip(population, self.min_weight, self.max_weight)

        # Normalize to sum to 1
        population = population / population.sum(axis=1, keepdims=True)

        logger.debug(f"Initialized population of {self.population_size} individuals")
        return population

    def calculate_fitness(
        self,
        population: np.ndarray,
        fitness_func: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """
        Calculate fitness for each individual in population

        Args:
            population: Population matrix
            fitness_func: Function that takes weights and returns fitness score

        Returns:
            Fitness scores array
        """
        fitness_scores = np.array([fitness_func(individual) for individual in population])
        return fitness_scores

    def tournament_selection(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """
        Select individuals using tournament selection

        Args:
            population: Population matrix
            fitness_scores: Fitness scores
            n_select: Number of individuals to select

        Returns:
            Selected individuals
        """
        selected = []

        for _ in range(n_select):
            # Random tournament
            tournament_idx = np.random.choice(
                self.population_size, size=self.tournament_size, replace=False
            )
            tournament_fitness = fitness_scores[tournament_idx]

            # Select winner (highest fitness)
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])

        return np.array(selected)

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents

        Uses blend crossover (BLX-α) which creates children by blending parent genes

        Args:
            parent1: First parent weights
            parent2: Second parent weights

        Returns:
            Tuple of two children
        """
        if np.random.random() > self.crossover_rate:
            # No crossover - return parents
            return parent1.copy(), parent2.copy()

        # BLX-α crossover (α = 0.5)
        alpha = 0.5

        child1 = np.zeros(self.n_assets)
        child2 = np.zeros(self.n_assets)

        for i in range(self.n_assets):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val

            # Blend with random value in extended range
            child1[i] = np.random.uniform(min_val - alpha * range_val, max_val + alpha * range_val)
            child2[i] = np.random.uniform(min_val - alpha * range_val, max_val + alpha * range_val)

        # Ensure bounds and normalization
        child1 = self._repair_solution(child1)
        child2 = self._repair_solution(child2)

        return child1, child2

    def mutate(self, individual: np.ndarray, generation: int) -> np.ndarray:
        """
        Mutate an individual

        Uses adaptive mutation rate that decreases over generations

        Args:
            individual: Individual to mutate
            generation: Current generation (for adaptive mutation)

        Returns:
            Mutated individual
        """
        # Adaptive mutation rate (decreases over time)
        adaptive_rate = self.mutation_rate * (1 - generation / self.n_generations)

        mutated = individual.copy()

        for i in range(self.n_assets):
            if np.random.random() < adaptive_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, 0.1)
                mutated[i] = mutated[i] + mutation

        # Repair solution to satisfy constraints
        mutated = self._repair_solution(mutated)

        return mutated

    def _repair_solution(self, weights: np.ndarray) -> np.ndarray:
        """
        Repair a solution to satisfy constraints

        Args:
            weights: Weights to repair

        Returns:
            Repaired weights
        """
        # Clip to bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Normalize to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # All weights are zero - reset to equal weight
            weights = np.ones(self.n_assets) / self.n_assets

        return weights

    def calculate_diversity(self, population: np.ndarray) -> float:
        """
        Calculate population diversity (standard deviation of weights)

        Args:
            population: Population matrix

        Returns:
            Diversity metric
        """
        diversity = np.std(population, axis=0).mean()
        return float(diversity)

    def evolve_generation(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray,
        generation: int,
    ) -> np.ndarray:
        """
        Evolve population for one generation

        Args:
            population: Current population
            fitness_scores: Fitness scores
            generation: Current generation number

        Returns:
            New population
        """
        new_population = []

        # Elitism - preserve best individuals
        elite_idx = np.argsort(fitness_scores)[-self.n_elites :]
        elites = population[elite_idx]
        new_population.extend(elites)

        # Generate offspring to fill remaining population
        n_offspring = self.population_size - self.n_elites

        while len(new_population) < self.population_size:
            # Select parents
            parents = self.tournament_selection(population, fitness_scores, 2)
            parent1, parent2 = parents[0], parents[1]

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutate(child1, generation)
            child2 = self.mutate(child2, generation)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return np.array(new_population)

    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        verbose: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run genetic algorithm optimization

        Args:
            fitness_func: Fitness function that takes weights and returns score
                         (higher is better)
            verbose: Print progress

        Returns:
            Tuple of (best_weights, optimization_info)
        """
        logger.info(f"Starting genetic algorithm optimization for {self.n_generations} generations")

        # Initialize
        population = self.initialize_population()
        best_individual = None
        best_fitness = -np.inf

        # Evolution loop
        for generation in range(self.n_generations):
            # Calculate fitness
            fitness_scores = self.calculate_fitness(population, fitness_func)

            # Track best individual
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            # Track statistics
            avg_fitness = np.mean(fitness_scores)
            diversity = self.calculate_diversity(population)

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)

            # Log progress
            if verbose and (generation % 10 == 0 or generation == self.n_generations - 1):
                logger.info(
                    f"Gen {generation:3d}: Best={best_fitness:.6f}, "
                    f"Avg={avg_fitness:.6f}, Diversity={diversity:.6f}"
                )

            # Evolve to next generation
            if generation < self.n_generations - 1:
                population = self.evolve_generation(population, fitness_scores, generation)

        # Prepare results
        results = {
            "best_fitness": float(best_fitness),
            "final_avg_fitness": float(self.avg_fitness_history[-1]),
            "final_diversity": float(self.diversity_history[-1]),
            "n_generations": self.n_generations,
            "convergence_history": {
                "best_fitness": self.best_fitness_history,
                "avg_fitness": self.avg_fitness_history,
                "diversity": self.diversity_history,
            },
        }

        logger.success(f"Genetic algorithm complete: Best fitness = {best_fitness:.6f}")

        return best_individual, results


def create_sharpe_fitness(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
) -> Callable[[np.ndarray], float]:
    """
    Create fitness function for maximizing Sharpe ratio

    Args:
        expected_returns: Expected returns vector
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate

    Returns:
        Fitness function
    """

    def fitness(weights: np.ndarray) -> float:
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        if portfolio_vol < 1e-8:
            return -np.inf

        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        return sharpe

    return fitness


def create_return_fitness(
    expected_returns: np.ndarray,
    max_volatility: float,
    cov_matrix: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """
    Create fitness function for maximizing return with volatility constraint

    Args:
        expected_returns: Expected returns vector
        max_volatility: Maximum allowed volatility
        cov_matrix: Covariance matrix

    Returns:
        Fitness function
    """

    def fitness(weights: np.ndarray) -> float:
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        # Penalty for exceeding volatility constraint
        if portfolio_vol > max_volatility:
            penalty = 1000 * (portfolio_vol - max_volatility)
            return portfolio_return - penalty

        return portfolio_return

    return fitness


def create_multiobjective_fitness(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> Callable[[np.ndarray], float]:
    """
    Create multi-objective fitness balancing return and risk

    Fitness = Return - risk_aversion * Variance

    Args:
        expected_returns: Expected returns vector
        cov_matrix: Covariance matrix
        risk_aversion: Risk aversion parameter (higher = more risk averse)

    Returns:
        Fitness function
    """

    def fitness(weights: np.ndarray) -> float:
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights @ cov_matrix @ weights

        # Mean-variance utility
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        return utility

    return fitness


def genetic_portfolio_optimization(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    objective: str = "sharpe",
    risk_free_rate: float = 0.02,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    population_size: int = 100,
    n_generations: int = 100,
    **kwargs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convenience function for genetic algorithm portfolio optimization

    Args:
        expected_returns: Expected returns vector
        cov_matrix: Covariance matrix
        objective: Optimization objective ('sharpe', 'return', 'utility')
        risk_free_rate: Risk-free rate
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        population_size: Size of population
        n_generations: Number of generations
        **kwargs: Additional arguments for GeneticPortfolioOptimizer

    Returns:
        Tuple of (optimal_weights, results_dict)

    Example:
        >>> expected_returns = np.array([0.10, 0.12, 0.08])
        >>> cov_matrix = np.array([[0.04, 0.01, 0.01],
        ...                         [0.01, 0.05, 0.02],
        ...                         [0.01, 0.02, 0.03]])
        >>> weights, results = genetic_portfolio_optimization(
        ...     expected_returns, cov_matrix, objective='sharpe'
        ... )
    """
    n_assets = len(expected_returns)

    # Create fitness function based on objective
    if objective == "sharpe":
        fitness_func = create_sharpe_fitness(expected_returns, cov_matrix, risk_free_rate)
    elif objective == "utility":
        risk_aversion = kwargs.pop("risk_aversion", 1.0)
        fitness_func = create_multiobjective_fitness(expected_returns, cov_matrix, risk_aversion)
    elif objective == "return":
        max_volatility = kwargs.pop("max_volatility", 0.15)
        fitness_func = create_return_fitness(expected_returns, max_volatility, cov_matrix)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Initialize optimizer
    optimizer = GeneticPortfolioOptimizer(
        n_assets=n_assets,
        population_size=population_size,
        n_generations=n_generations,
        min_weight=min_weight,
        max_weight=max_weight,
        **kwargs,
    )

    # Optimize
    optimal_weights, results = optimizer.optimize(fitness_func, verbose=True)

    # Add portfolio metrics to results
    portfolio_return = optimal_weights @ expected_returns
    portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

    results.update(
        {
            "portfolio_return": float(portfolio_return),
            "portfolio_volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe),
            "objective": objective,
        }
    )

    return optimal_weights, results
