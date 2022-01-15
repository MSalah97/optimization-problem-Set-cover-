import random
import typing
import numpy as np

def greedy_set_cover(cover, cost):
    act_sirens = []
    tot_cost = 0
    covered = set()
    areas = set(range(0, 30))
    d = []
    for i in range(len(cover)):
        temp = 0
        for j in range(len(cover[i])):    # calculate the dj per every siren,the number of the covered areas
            if cover[i][j] == 1:
                temp += 1
        d.append(temp)

    
    cost_per_place = []          
    for k in range(len(cover)):             # calculate the ratio Cj(cost per siren)/Dj(areas covered)
        temp2 = cost[k]/d[k]
        cost_per_place.append(temp2)
    
    while covered != areas:                 # choose the siren that has the minimum ratio,and add its covered areas
        ind = cost_per_place.index(min(cost_per_place))
        for i in range(len(cover[ind])):                    
            if cover[ind][i] == 1:              
                covered.add(i)
        tot_cost += cost[ind]
        act_sirens.append(ind+ 1)
        cost_per_place[ind] = 999    # to not choose the selected siren again

    return act_sirens, tot_cost    

''' def local_search(covered, cover, cost, ind):
    act_sirens = []
    tot_cost = 0
    areas = set(range(0, 30))
    if covered == areas:                # stopping condition in our case ,the whole areas will be covered
        return act_sirens,tot_cost

    #add the areas covered by the new siren to the covered set
    for i in range(30):                    
        if cover[ind][i] == 1:              
            covered.add(i)
    tot_cost += cost[ind]

    act_sirens.append(ind+1)  
    
    #evaluate every neighbour of the current siren and choose the best one then iterate from it
    neighbour = []
    for i in range(len(cover)):                 
        temp = 0
        for j in range(len(cover[i])):    
            if cover[i][j] == 1 and not(j in covered):
                temp += 1
        neighbour.append(temp)

    cost_per_place = []          
    for k in range(len(cover)):             
        temp2 = 999         #for not consider the current siren again
        if neighbour[k] != 0:
            temp2 = cost[k]/neighbour[k]
        cost_per_place.append(temp2)

    # find the siren with minimum ratio
    best_neighbor = cost_per_place.index(min(cost_per_place))   

    a ,c  = local_search(covered,cover,cost,best_neighbor)
    
    return act_sirens + a, tot_cost + c'''''


def grasp(covered, cover, cost):  #instead of starting from random point , start from a greedy result
    d = []
    for i in range(len(cover)):                 
        temp = 0
        for j in range(len(cover[i])):    
            if cover[i][j] == 1:
                temp += 1
        d.append(temp)

    cost_per_place = []          
    for k in range(len(cover)):             
        temp2 = cost[k]/d[k]
        cost_per_place.append(temp2)

    # find the siren with minimum ratio
    greedy_start = cost_per_place.index(min(cost_per_place)) 

    return local_search(covered, cover, cost, greedy_start)


def iterated_local_search(covered, cover, cost):  # in order to escape from local minima
    best_sirens = []
    best_cost = 999
    for i in range(0, len(cost)):
        a, c = local_search(covered.copy(), cover.copy(), cost.copy(), i)
        if c < best_cost and c != 0:
            best_cost = c
            best_sirens = a

    return best_sirens, best_cost


def create_gene(): # 1st : genetic rap. of a solution
    temp = 0
    gene=[]
    n = random.randint(0, 12)

    while temp != 4:
        if n not in gene:
            gene.append(n)
            temp += 1
            n = random.randint(0, 12)
        else:
            n = random.randint(0, 12)
    return gene


def create_population(no_gene): # 2nd : function to generate new solution
    population=[]
    while no_gene > 0:
        gene = create_gene()
        population.append(gene)
        no_gene -= 1
    return population


def fitness_func(gene): # 3rd : fitness function to evaluate solution
    areas = coverage.copy()
    cost = cost_sirens.copy()
    covered=set()
    to_cover = set(range(0,30))
    value = 0

    for i in gene:
        for j in range(len(areas[i])):
            if areas[i][j] == 1:
                covered.add(j)
    if covered == to_cover:
        for i in gene:
            value += cost[i]
        return value
    else:
        return 0


def sort_pop(population):
    return population.sort(key=fitness_func)

def selection_func(population): # 4th : selection function to create the next generation
    parent1=random.choice(population)
    parent2=random.choice(population)
    return parent1,parent2


def single_point_crossover(parent1,parent2): #5th:single crossover to generate a new solution

    return parent1[0:2]+parent2[2:],parent2[0:2]+parent1[2:]

def mutation_func(gene):
    temp=0
    n=random.randint(0,12)
    while temp!=1:
        if n not in gene:
            gene[1] = n
            temp += 1
        else:
            n = random.randint(0, 12)
    return gene

def run_evolution(generation_limit):
    num_gene=20
    population = create_population(num_gene)

    for i in range(generation_limit):

        sort_pop(population)

        if fitness_func(population[0]):
            break

        next_generation = population[0:2]  # elitism

        for j in range(int(len(population) / 2)):
            parent1,parent2 = selection_func(population)
            offspring_a, offspring_b = single_point_crossover(parent1, parent2)
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
        population = next_generation

    return population

def genetic_alg(generation_limit,cost):
    population = run_evolution(generation_limit)
    act_sir=population[0]
    tot_cost=0
    for i in act_sir:
        tot_cost += cost[i]

    return  act_sir,tot_cost




cost_sirens = [78, 30, 76, 69, 75, 67, 44, 57, 35, 87, 65, 36, 99]


# coverage areas for every siren
coverage = [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]]


print("\n##  solution by algorithm  ##\n")
active_sirens, total_cost = greedy_set_cover(coverage.copy(), cost_sirens.copy())
print(f'\t(greedy search)\nthe active sirens are: {active_sirens}\nthe the cost will be: {total_cost}')
print()

active_sirens, total_cost = local_search(set(), coverage.copy(), cost_sirens.copy(),random.randint(0,11))
print(f'\t(local search)\nthe active sirens are: {active_sirens}\nthe the cost will be: {total_cost}')
print()

active_sirens, total_cost = grasp(set(), coverage.copy(), cost_sirens.copy())
print(f'\t(grasp search)\nthe active sirens are: {active_sirens}\nthe the cost will be: {total_cost}')
print()

active_sirens, total_cost = iterated_local_search(set(), coverage.copy(), cost_sirens.copy())
print(f'\t(for iterated local search)\nthe active sirens are: {active_sirens}\nthe the cost will be: {total_cost}')
print()

active_sirens, total_cost = genetic_alg(10, cost_sirens.copy())
print(f'\t(genetic algorithm)\nthe active sirens are: {active_sirens}\nthe the cost will be: {total_cost}')
print()


def simulated_annealing(areas, cost): # The method will perform the simulated annealing algorithm and it will keep track of the best found solution

    temperature = 100
    best_cost = 999
    best_sirens = []
    a, c = local_search(set(),areas.copy(), cost.copy(),random.randint(0,12))
    while temperature > 0:
        alfa = c - best_cost
        if (alfa < 0) or random.randint(0, 1) < np.exp(-1*alfa/temperature):
            best_cost = c
            best_sirens = a
        temperature = 0.5 * temperature
    return best_sirens,best_cost

active_sirens, total_cost = simulated_annealing(coverage.copy(), cost_sirens.copy())
print(f'\t(for simulated_annealing)\nthe active sirens are: {active_sirens}\nthe the cost will be: {total_cost}')
print()
