import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from collections import namedtuple


Individual = namedtuple("Individual", "state fitness min_range max_range")

#Used to checks if the diagonal clash has already occured.
def no_repeats(col, row, repeats):
    for item in repeats:
        if item[0] == col and item[1] == row:
            return False
    return True

#Returns the number of unique clashes that occur in the upward_diagonal direction
def check_diagonal_up(state):
    num_clashes = 0
    repeats = []
    #Check upward right
    for i in range(0, 8):
        row_pos = state[i]
        for j in range(i+1, 8):
            row_pos += 1
            if state[j] == row_pos and no_repeats(j, row_pos, repeats):
                repeats.append((j, row_pos))
                num_clashes += 1
    #Check upward left
    for i in range(7, -1, -1):
        row_pos = state[i]
        for j in range(i - 1, -1, -1):
            row_pos += 1
            if state[j] == row_pos and no_repeats(j, row_pos, repeats):
                repeats.append((j, row_pos))
                num_clashes += 1
    return num_clashes

#Returns the number of unique clashes that occur in the downward_diagonal direction
def check_diagonal_down(state):
    num_clashes = 0
    repeats = []
    #Check downward right
    for i in range(0, 8):
        row_pos = state[i]
        for j in range(i+1, 8):
            row_pos -= 1
            if state[j] == row_pos and no_repeats(j, row_pos, repeats):
                repeats.append((j, row_pos))
                num_clashes += 1
    #Check downward left
    for i in range(7, -1, -1):
        row_pos = state[i]
        for j in range(i-1, -1, -1):
            row_pos -= 1
            if state[j] == row_pos and no_repeats(j, row_pos, repeats):
                repeats.append((j, row_pos))
                num_clashes += 1
    return num_clashes

#Returns the number of non attacking queens for a given board state.
def fitness_function(individual):
    num_clashes = 0
    num_clashes += abs(len(individual) - len(np.unique(individual)))  #queens in the same row
    num_clashes += check_diagonal_up(individual)  #queens clashing diagonal upward_direction
    num_clashes += check_diagonal_down(individual)  #queens clashing diagonal down_direction
    return 28 - num_clashes

#Takes in two parents and crosses them over at a random index in the string.
#This creates two new children, which are then returned.
def crossover(state_1, state_2):
    cross_point = np.random.randint(0, 8)
    mutation_pct_1 = np.random.uniform()
    mutation_pct_2 = np.random.uniform()

    pre_cross_point_1 = np.array(state_1[0:cross_point]).tolist()
    post_cross_point_1 = np.array(state_1[cross_point:]).tolist()
    pre_cross_point_2 = np.array(state_2[0:cross_point]).tolist()
    post_cross_point_2 = np.array(state_2[cross_point:]).tolist()

    child_1_state = pre_cross_point_1 + post_cross_point_2
    child_2_state = pre_cross_point_2 + post_cross_point_1

    #Chance for mutation to occur
    if mutation_pct_1 <= .05:
        mut_point = np.random.randint(0, 8)
        child_1_state[mut_point] = np.random.randint(1, 9)

    if mutation_pct_2 <= .05:
        mut_point = np.random.randint(0, 8)
        child_2_state[mut_point] = np.random.randint(1, 9)

    return child_1_state, child_2_state

#Creates a popultion of individuals of a given popultion size.
#Returns the generated population, sorted by an individuals fitness
def generate_pop(pop_size):
    population = []
    for i in range(0, pop_size):
        state = np.random.randint(1, 9, 8)
        fitness = fitness_function(state)
        individual = Individual(state, fitness, 0.0, 1.0)
        population.append(individual)
        population = sorted(population, key=lambda x: x[1], reverse=True)
    return population

#Returns total fitness of the given population
def total_fitness(population):
    tot_fit = 0
    for item in population:
        tot_fit += item[1]
    return tot_fit

#Updates the probability range for each individual in the popultion.
#Range is determined based on the individuals fitness proportional to the total fitness of the popultion.
def update_prob_range(population):
    prob = 0
    tot_fit = total_fitness(population)
    updated_population = []
    for item in population:
        state = item.state
        fitness = item.fitness
        min_range = prob
        max_range = prob + (fitness / tot_fit)
        prob += fitness/tot_fit
        individual = Individual(state, fitness, min_range, max_range)
        updated_population.append(individual)

    return updated_population

#Returns the individual where the given probability falls within its range of probability
def get_parent(population, prob):
    for item in population:
        if item.min_range <= prob and item.max_range > prob:
            return item
    #Just in case nothing is found choose a random
    rand_index = np.random.randint(0, len(population)-1)
    return population[rand_index]

#Simulates one generation of reproduction among randomly chosen parents.
#Returns a new population with children of chosen parents equal in size to the orginal population size.
def reproduce(population, pop_size):
    new_population = []

    while len(new_population) < pop_size:

        prob_1 = np.random.uniform()
        prob_2 = np.random.uniform()

        parent_1 = get_parent(population, prob_1)
        parent_2 = get_parent(population, prob_2)
        child_1_state, child_2_state = crossover(parent_1.state, parent_2.state)
        child_1 = Individual(child_1_state, fitness_function(child_1_state), 0, 1)
        child_2 = Individual(child_2_state, fitness_function(child_2_state), 0, 1)
        new_population.append(child_1)
        new_population.append(child_2)

    new_population = update_prob_range(new_population)
    return new_population

#Simulate the genetic algortithm
#Displays average fitness of each generation
#Outputs ten individuals from unique generations
def genetic_algorithm(pop_size, num_iterations):
    init_population = generate_pop(pop_size)
    init_population = update_prob_range(init_population)
    best_individual = []
    average_fitness = []
    new_population = cp.deepcopy(init_population)
    for i in range(0, num_iterations):
        new_population = cp.deepcopy(reproduce(new_population, pop_size))
        curr_population = cp.deepcopy(sorted(new_population, key=lambda x: x[1], reverse=True))

        average_fit_individ = total_fitness(curr_population) / pop_size
        average_fitness.append(average_fit_individ)
        best_indiv = curr_population[0]
        best_individual.append(best_indiv)

    print("Pop Size:", pop_size, "Number Iterations:", num_iterations)

    for i in range(0, len(best_individual)):
        if ((i/num_iterations)*100) % 10 == 0:
            print("Best Individual in Generation ",i,":\n",best_individual[i])

    print("Best Individual in Last Generation ", num_iterations, ":\n", best_individual[num_iterations-1])

    plt.plot(average_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness Score")
    plt.suptitle("Population Size: %d\n Number Iterations: %d" %(pop_size, num_iterations))
    plt.show()


#Run the genetic algorithm
num_iterations = 100
pop_size = 2500
genetic_algorithm(pop_size, num_iterations)
