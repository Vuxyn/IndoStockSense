"""PySpark helpers for parallel GA fitness evaluation."""


def evaluate_population_parallel(sc, population, evaluator, num_slices: int = 4):
    """Evaluate GA population in parallel using Spark RDD map."""
    pop_rdd = sc.parallelize(population, numSlices=num_slices)
    fitness_rdd = pop_rdd.map(lambda individual: evaluator(individual))
    return fitness_rdd.collect()
