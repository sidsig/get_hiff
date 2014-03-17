import random,copy,os,pickle

from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from rbm import RBM
from optimizers import sgd_optimizer

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import theano
from custom_dataset import SequenceDataset
import pdb

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class KnapsackData(object):
    """docstring for Knapsack"""
    def __init__(self, name, knapsacks, items, values, capacities, constraints):
        super(KnapsackData, self).__init__()
        self.name = name
        self.knapsacks = knapsacks
        self.items = items
        self.values = values
        self.capacities = capacities
        self.constraints = constraints

class Genotypes(object):
    def __init__(self, min=True):
        super(Genotypes, self).__init__()
        self.genotype_array = []
        self.keys = {}
        self.min = min
        self.window = 100

    def add_genotypes(self,genotypes,uniques = False):
        if len(self.genotype_array) >= self.window:
            removing = self.genotype_array.pop()
            if uniques:
                hashed_g = str(self.genotype_array)
                for g in removing:
                    hashed_genotype = str(g)
                    if hashed_genotype not in hashed_g:
                        del self.keys[hashed_genotype]
        unique_genotypes = []
        for g in genotypes:
            if uniques:
                if str(g.genotype) not in self.keys:
                    unique_genotypes.append(g)
                    self.keys[str(g.genotype)] = 1
            else:
                unique_genotypes.append(g)
        self.genotype_array.insert(0,unique_genotypes)

    def flatten(self):
        return [item for sublist in self.genotype_array for item in sublist]

    def top_x_percent(self,x=0.2):
        s = self.flatten()
        s = sorted(s,key=lambda member:member.fitness)
        if self.min == False:
            s.reverse()
        s = [_.genotype for _ in s]
        self.top_x = s[0:int(len(s)*x)]
        return self.top_x

    def get_and_save_top_x(self,x=0.2,path="experiments",experiment=0,generation=0):
        top_x = self.top_x_percent(x)
        # self.top_x_genotypes_to_file(top_x,path=path,experiment=experiment,generation=generation)


    def top_x_genotypes_to_file(self,top_20,path="experiments",experiment=0,generation=0):
        np.savetxt("{0}_{1}/top_genotypes_{2}.dat".format(path,experiment,generation),top_20)
        np.savetxt("{0}_{1}/top_genotypes_fitnesses_{2}.dat".format(path,experiment,generation),[f.fitness.values[0] for f in top_20])

class Individual(object):
    def __init__(self):
        super(Individual, self).__init__()
        self.genotype = None
        self.fitness = None
        self.normalised_fitness = 0
        
class ES(object):
    def __init__(self,knapsack_file="weing1.pkl"):
        super(ES, self).__init__()
        # GA stuff
        self.generations = 100
        self.knapsack = pickle.load(open(knapsack_file))
        print "k:",self.knapsack
        self.N = int(self.knapsack.items)
        # RMB stuff
        self.RBM = RBM(n_visible=self.N,n_hidden=50) 
        self.sample_RBM()

        # Stats stuff
        self.population_snapshots = []
        self.genotypes_history = Genotypes(min=False)

    def create_individual(self,N):
        I = Individual()
        I.genotype = [random.choice([0,1]) for i in range(N)]
        I.fitness = 0
        return I

    def fitness_function(self,individual,knapsack=None):
        weights = []
        for i,c in enumerate(knapsack.capacities):
            weights.append(np.sum(np.array(knapsack.constraints[i])*individual.genotype))
        over = 0
        for i,w in enumerate(weights):
            if w > knapsack.capacities[i]:
                over += (w - knapsack.capacities[i])
        if over > 0:
            return -over
        else:
            return np.sum(np.array(knapsack.values)*individual.genotype)

    def evaluate_population(self,population,params=None):
        for p in population:
            p.fitness = self.fitness_function(p,params)

    def normalise_fitnesses(self,population):
        max_fitness = np.max([p.fitness for p in population])
        min_fitness = np.min([p.fitness for p in population])
        for p in population:
            p.normalised_fitness = (p.fitness + min_fitness)/(min_fitness+max_fitness)

    def offspring_from_sample(self,individual_to_copy):
        individual = copy.deepcopy(individual_to_copy)
        individual_genome = np.array(individual.genotype).reshape(1,-1)
        output = self.sample_from_RBM(np.array(individual_genome))
        # print "output:",output
        individual.genotype[:] = output[0][:]
        return individual

    def train_RBM(self,k=20,lr=0.1):
        train_data = self.genotypes_history.top_x_percent()
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        inputs,params,cost,monitor,updates,consider_constant = self.RBM.build_RBM(k=k)
        sgd_optimizer(params,[inputs],cost,train_set,updates_old=updates,monitor=monitor,
                      consider_constant=[consider_constant],lr=0.1,num_epochs=10)

    def sample_RBM(self,k=20):
        v,v_sample,updates = self.RBM.sample_RBM(k=k)
        self.sample_from_RBM = theano.function([v],v_sample,updates=updates)

    def run_1_plus_1(self, path= "", experiment = 0):
        random.seed(random.uniform(0,1000000))
        print("Start of evolution")
        parent = self.create_individual(self.N)
        # Evaluate the parent
        parent.fitness = self.fitness_function(parent,self.knapsack)
        self.genotypes_history.add_genotypes([parent])
        self.genotypes_history.get_and_save_top_x(1.0)
        self.train_RBM()

        # Begin the evolution
        for g in range(self.generations):
            print("-- Generation %i --" % (g + 1))
            offspring = self.offspring_from_sample(parent)
            offspring.fitness = self.fitness_function(parent,self.knapsack)
            print "parent_fitness:",parent.fitness
            print "offspring_fitness:",offspring.fitness
            if offspring.fitness > parent.fitness:
                print "offspring replacing parent"
                parent = offspring
            self.genotypes_history.add_genotypes([offspring])
            self.genotypes_history.get_and_save_top_x(1.0)
            self.train_RBM()
        print("-- End of (successful) evolution --")
        return parent

    def run_mu_plus_lambda(self, path= "", experiment = 0):
        population_size = 50
        random.seed(random.uniform(0,1000000))
        print("Start of evolution")
        population = [self.create_individual(self.N) for i in range(population_size)]
        # Evaluate the population
        self.evaluate_population(population,self.knapsack)
        self.genotypes_history.add_genotypes(population)
        self.genotypes_history.get_and_save_top_x(1.0)
        self.train_RBM()

        # Begin the evolution
        for g in range(self.generations):
            print("-- Generation %i --" % (g + 1))
            offspring = []
            for ind in population:
                offspring.append(self.offspring_from_sample(ind))
            self.evaluate_population(offspring,self.knapsack)
            self.genotypes_history.add_genotypes(offspring)
            self.genotypes_history.get_and_save_top_x(1.0)
            self.train_RBM()
            new_population = []
            population = population + offspring
            while len(new_population) < population_size:
                # tournament selection on combined population
                a = int(len(population) * random.random())
                b = int(len(population) * random.random())
                while a == b:
                    b = int(len(population) * random.random())
                if population[a].fitness > population[b].fitness:
                    new_population.append(population.pop(a))
                else:
                    new_population.append(population.pop(b))
            population = new_population
            print "average fitness:",np.mean([p.fitness for p in population])
            print "max fitness:",np.max([p.fitness for p in population])
            print "min fitness:",np.min([p.fitness for p in population])
        print("-- End of (successful) evolution --")
        return parent

class RandomSearch(ES):
    """docstring for RandomSearch"""
    def __init__(self,knapsack_file="weing1.pkl"):
        super(RandomSearch, self).__init__(knapsack_file="weing1.pkl")
        self.solutions = []
        print knapsack_file

    def run(self,experiment_name,trials):
        mean = []
        min = []
        max = []
        for i in range(trials):
            individual = self.create_individual(self.N)
            individual.fitness = self.fitness_function(individual,self.knapsack)
            self.solutions.append(individual)
            if i% 100 == 0:
                all_fitnesses = [s.fitness for s in self.solutions]
                mean.append(np.mean(all_fitnesses))
                max.append(np.max(all_fitnesses))
                min.append(np.min(all_fitnesses))
        self.solutions = sorted(self.solutions,key = lambda s:s.fitness)
        self.solutions.reverse()
        all_fitnesses = [s.fitness for s in self.solutions]
        print "max", np.max(all_fitnesses)
        print "min", np.min(all_fitnesses)
        print "mean", np.mean(all_fitnesses)
        ensure_dir("results/random/hard_knapsack/")
        np.savetxt("results/random/hard_knapsack/means_{0}".format(experiment_name),mean)
        np.savetxt("results/random/hard_knapsack/max_{0}".format(experiment_name),max)
        np.savetxt("results/random/hard_knapsack/min_{0}".format(experiment_name),min)



        
if __name__ == "__main__":
    # e = ES(knapsack_file="weing1.pkl")
    # e.run_mu_plus_lambda()
    for i in range(0,10):
        r = RandomSearch(knapsack_file="weing8.pkl")
        r.run(i,200000)
