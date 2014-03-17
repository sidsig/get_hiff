import random,pickle,copy,bisect
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from rbm import *
from denoising_autoencoder import dA
from deep_da import Deep_dA
import theano
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
from numpy import array as ar
import pdb
import distance
import os,sys
from ga import KnapsackData
from hiff import HIFF
from max_sat import MAXSAT

DO_CACHING = True
if DO_CACHING:
    def cached_string(f):
        """
        This is a decorator to make sure
        the should still be active
        """
        def wrapped(self,*args):
            string = self.i_to_s(args[0])
            if string in self.fitness_cache:
                return self.fitness_cache[string]
            else:
                return f(self,*args,cache=True)
        return wrapped
else:
    def cached_string(f):
        """
        This is a decorator to make sure
        the should still be active
        """
        def wrapped(self,*args):
            return f(self,*args,cache=False)
        return wrapped

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class LeftOnes(object):
    """docstring for LeftOnes"""

    def __init__(self,corruption_level=0.2):
        super(LeftOnes, self).__init__()
        self.dA = Deep_dA(n_visible=20,n_hidden=[50])
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        self.fitness_cache = {}
    def weighted_random_choice(self,population,fitness_sum,cumulative_dist):
        pick = random.uniform(0, fitness_sum)
        index = bisect.bisect_right(cumulative_dist,pick)
        if index > len(population) - 1:
            index = len(population) - 1
        return copy.deepcopy(population[index])

    def get_new_population_rw(self,population, fitnesses=None, pop_size=None):
        if pop_size == None:
            pop_size = len(population)
        if fitnesses == None:
            fitnesses = self.fitness_many(population)
        sum_fitnesses = sum(fitnesses)
        cumulative_dist = []
        current = 0.0
        for chromosome in population:
            current += self.fitness(chromosome)
            cumulative_dist.append(current)
        new_population = [self.weighted_random_choice(population,sum_fitnesses,cumulative_dist)
                            for p in range(pop_size)]
        return population

    # def get_new_population_rw(self,population, fitnesses=None, pop_size=None):
    #     if pop_size == None:
    #         pop_size = len(population)
    #     if fitnesses == None:
    #         fitnesses = self.fitness_many(population)
    #     # print ar(fitnesses)
    #     # print (ar(fitnesses)+0.0)/np.sum(fitnesses)
    #     # print np.sum(ar(fitnesses)/np.sum(fitnesses))
    #     choices = np.random.choice(pop_size,pop_size,p=(ar(fitnesses)+0.0)/np.sum(fitnesses))
    #     new_population = [population[int(p)]
    #                         for p in choices]
    #     return population

    def cache_fitness(self,string,fitness):
        self.fitness_cache[self.i_to_s(string)] = fitness

    def i_to_s(self,string):
        return str(list(string))

    @cached_string
    def fitness(self,string):
        fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
        return fitness

    def fitness_many(self,strings):
        return [self.fitness(s) for s in strings]

    def generate_random_string(self,l=20):
        return [random.choice([0,1]) for i in range(l)]

    def get_good_strings(self,strings,lim=20):
        # print strings[0:5]
        fitnesses = [self.fitness(s) for s in strings]
        # print fitnesses[0:10]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        return [strings[i] for i in sorted_fitnesses[0:lim]],[fitnesses[k] for k in sorted_fitnesses[0:lim]]

    def generate_good_strings(self,x=1000,l=20,lim=20):
        strings = [[random.choice([0,1]) for i in range(l)] for _ in range(x)]
        fitnesses =  [self.fitness(s) for s in strings]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        return strings,[strings[i] for i in sorted_fitnesses[0:lim]]

    def train_dA(self,data,corruption_level=0.2,num_epochs=200,lr=0.1,output_folder="",iteration=0):
        train_data = data
        #print_var = theano.printing.Print()(self.dA.input)
        #print_fn = theano.function([self.dA.input],print_var)
        # pdb.set_trace()
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,lr=lr,
                      num_epochs=num_epochs,save=False,output_folder=output_folder,iteration=iteration)

    def train_RBM(self,data,num_epochs=200,lr=0.1,output_folder="",):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.RBM.params,[self.RBM.input],self.RBM.cost,train_set,consider_constant=self.RBM.consider_constant,updates=self.RBM.updates,save=True)

    def build_sample_RBM(self,):
        self.sample_RBM = theano.function([self.RBM.input],self.RBM.v_sample)

    def build_sample_dA_2(self,k=20):
        samples,updates = self.dA.build_sampler(k=k)
        self.sample_dA = theano.function([self.dA.input],samples,updates=updates)

    def build_sample_dA(self):  
        self.sample_dA = theano.function([self.dA.input],self.dA.sample)

    def calculate_changes_in_fitness(self,population,number_of_trials):
        original_fitnesses = ar(self.fitness_many(population))
        print original_fitnesses.shape
        sample = [self.sample_dA([i]) for i in population]
        # print sample.shape
        sample_fitnesses = ar(self.fitness_many([j for j in sample]))
        # return original_fitnesses,sample,sample_fitnesses
        print sample_fitnesses.shape
        print sample_fitnesses[0:10]
        differences = sample_fitnesses - original_fitnesses
        distances = [[distance.hamming(population[k],sample[k]) for k in range(len(sample))]]
        # pdb.set_trace()
        for i in range(number_of_trials):
            print "trial:",i
            new_sample = [self.sample_dA([j]) for j in population]
            new_sample_fitnesses = ar(self.fitness_many([j for j in new_sample]))
            new_difference = new_sample_fitnesses - original_fitnesses
            sample_fitnesses = np.vstack((sample_fitnesses,new_sample_fitnesses))
            differences = np.vstack((differences,new_difference))
            distances.append([distance.hamming(population[k],sample[k]) for k in range(len(sample))])
        return sample_fitnesses,differences,distances

    def get_statistics(self,population,sample,get_distances=False,original_fitnesses=False):
        if original_fitnesses:
            original_fitnesses = ar(self.fitness_many(population))
        sample_fitnesses = ar(self.fitness_many(sample))
        if original_fitnesses == False:
            original_fitnesses = sample_fitnesses
        if get_distances:
            differences = sample_fitnesses - original_fitnesses
            distances = [distance.hamming(population[k],sample[k]) for k in range(len(sample))]
        else:
            differences = []
            distances = []
        return original_fitnesses,sample_fitnesses,differences,distances

    def save_population(self,population,iteration):
        np.savetxt("population_{0}.dat".format(iteration),population)

    def save_training_data(self,tdata,iteration):
        np.savetxt("training_data_{0}.dat".format(iteration),tdata)

    def save_sampled_random_population(self,population,iteration):
        np.savetxt("random_sampled_population_{0}.dat".format(iteration),population)

    def save_pop_fitnesses(self,fitnesses,fname,iteration):
        np.savetxt("{0}_{1}.dat".format(fname,iteration),fitnesses)

    def save_random_population(self,population,iteration):
        np.savetxt("random_population_{0}.dat".format(iteration),population)

    def experiment(self,name,no_trials=10,corruption_level=0.2):
        ensure_dir("results/autoencoder/".format(name))
        all_strings,good_strings=self.generate_good_strings(10000)
        self.train_dA(ar(good_strings),corruption_level=corruption_level)
        original_fitnesses = self.fitness_many(all_strings)
        f,d,dist = self.calculate_changes_in_fitness(all_strings,no_trials)
        data = {
        "original":original_fitnesses,
        "fitnesses_sampled":f,
        "differences_in_fitness":d,
        "distances":dist,
        "no_trials":no_trials,
        "corruption_level":corruption_level,
        "all_strings":all_strings,
        "good_strings":good_strings
        }
        pickle.dump(data,open("results/autoencoder/{0}.pkl".format(name),"wb"))
        return data

    def iterative_algorithm(
        self,
        name,
        pop_size=100,
        genome_length=20,
        lim_percentage=20,
        lim=20,
        trials=10,
        corruption_level=0.2,
        num_epochs=50,
        lr = 0.1,
        online_training=False,
        pickle_data=False,
        max_evaluations=200000,
        save_data=False):
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        trials = max_evaluations/pop_size
        population_limit = lim
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA(n_visible=genome_length,n_hidden=50)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,population_limit)
        self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr,output_folder=results_path,iteration=0)
        # sampled_population = [self.sample_dA([i]) for i in self.get_new_population_rw(all_strings)]
        sampled_population = np.array(self.sample_dA(self.get_new_population_rw(all_strings)),"b")
        print "s:",sampled_population
        original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,sampled_population)
        data = {
        "original":original_fitnesses,
        "fitnesses_sampled":sample_fitnesses,
        "differences_in_fitness":differences,
        "distances":distances
        }
        if pickle_data:
            pickle.dump(data,open("results/autoencoder/{0}_0.pkl".format(name),"wb"))
        if save_data:
            self.save_population(sampled_population,0)
            self.save_training_data(good_strings,0)
            random_pop = [self.generate_random_string(genome_length) for z in range(1000)]
            print random_pop[0]
            sampled_r_pop = self.sample_dA(random_pop)
            self.save_sampled_random_population([r for r in sampled_r_pop],0)
            self.save_random_population(random_pop,0)
            self.save_pop_fitnesses(sample_fitnesses,"fitness_pop",0)
            self.save_pop_fitnesses(self.fitness_many(good_strings),"fitness_training_data",0)
        print "writing..."
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses),np.std(original_fitnesses)))
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
        print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        print "writing over"
        for iteration in range(0,trials):
            print "choosing pop:"
            population = self.get_new_population_rw(sampled_population,sample_fitnesses)
            print "choosing pop over"
            if online_training == False:
                print "building model..."
                self.dA = Deep_dA(n_visible=genome_length,n_hidden=50)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr,output_folder=results_path,iteration=iteration+1)
            print "sampling..."
            sampled_population = np.array(self.sample_dA(population),"b")
            print "sampling over"
            print "getting_statistics..."
            original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,sampled_population)
            data = {
            "original":original_fitnesses,
            "fitnesses_sampled":sample_fitnesses,
            "differences_in_fitness":differences,
            "distances":distances
            }
            print "statistics over"
            if pickle_data:
                pickle.dump(data,open("results/autoencoder/{0}_{1}.pkl".format(name,iteration+1),"wb"))
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
            if save_data:
                print "saving stuff..."
                self.save_population(sampled_population,iteration+1)
                self.save_training_data(good_strings,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(1000)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population([r for r in sampled_r_pop],iteration+1)
                self.save_random_population(random_pop,iteration+1)
                self.save_pop_fitnesses(sample_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()

class ACSKnapsack(LeftOnes):
    """docstring for ACSKnapsack"""
    def __init__(self, corruption_level=0.2,knapsack_file="weing1.pkl"):
        super(ACSKnapsack, self).__init__(corruption_level=corruption_level)
        self.knapsack = pickle.load(open(knapsack_file))

    def iterative_algorithm(
        self,
        name,
        pop_size=100,
        genome_length=20,
        lim_percentage=20,
        lim=20,
        trials=10,
        corruption_level=0.2,
        num_epochs=50,
        lr = 0.1,
        online_training=False,
        pickle_data=False,
        max_evaluations=200000,
        save_data=False,
        cross_rate=0.9):
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA(n_visible=genome_length,n_hidden=80)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,population_limit)
        self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr,output_folder=results_path,iteration=0)
        # sampled_population = [self.sample_dA([i]) for i in self.get_new_population_rw(all_strings)]
        fit_p_pop = self.get_new_population_rw(all_strings)
        sampled_population = np.array(self.sample_dA(fit_p_pop),"b")
        new_population = self.create_population_with_unif_cross(sampled_population,fit_p_pop,cross_rate)
        # print "s:",sampled_population
        original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,new_population)
        if save_data:
            self.save_population(sampled_population,0)
            self.save_training_data(good_strings,0)
            random_pop = [self.generate_random_string(genome_length) for z in range(1000)]
            print random_pop[0]
            sampled_r_pop = self.sample_dA(random_pop)
            self.save_sampled_random_population([r for r in sampled_r_pop],0)
            self.save_random_population(random_pop,0)
            self.save_pop_fitnesses(sample_fitnesses,"fitness_pop",0)
            self.save_pop_fitnesses(self.fitness_many(good_strings),"fitness_training_data",0)
        print "writing..."
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses),np.std(original_fitnesses)))
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
        print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        print "writing over"
        for iteration in range(0,trials):
            print "choosing pop:"
            population = self.get_new_population_rw(new_population,sample_fitnesses)
            population[0] = new_population[np.argmax(sample_fitnesses)]
            print "choosing pop over"
            if online_training == False:
                print "building model..."
                self.dA = Deep_dA(n_visible=genome_length,n_hidden=100)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=True)
            # return self.get_good_strings(population,population_limit,unique=True)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr,output_folder=results_path,iteration=iteration+1)
            print "sampling..."
            sampled_population = np.array(self.sample_dA(population),"b")
            new_population = self.create_population_with_unif_cross(sampled_population,population,cross_rate)
            new_population[0:1] = good_strings[0:1]
            print "sampling over"
            print "getting_statistics..."
            original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,new_population)
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(sample_fitnesses)])
            if save_data:
                print "saving stuff..."
                self.save_population(sampled_population,iteration+1)
                self.save_training_data(good_strings,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(1000)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population([r for r in sampled_r_pop],iteration+1)
                self.save_random_population(random_pop,iteration+1)
                self.save_pop_fitnesses(sample_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()

    @cached_string
    def fitness(self,string,cache=False):
        knapsack = self.knapsack
        weights = []
        for i,c in enumerate(knapsack.capacities):
            weights.append(np.sum(np.array(knapsack.constraints[i])*string))
        over = 0
        for i,w in enumerate(weights):
            if w > knapsack.capacities[i]:
                over += (w - knapsack.capacities[i])
        if over > 0:
            if cache:
                self.cache_fitness(string,-over)
            return -over
        else:
            _fitness = np.sum(np.array(knapsack.values)*string)
            if cache:
                self.cache_fitness(string,_fitness)
            return _fitness
    def create_population_with_unif_cross(self,sampled_population,population,cross_rate=0.1):
        new_population = []
        for i,p in enumerate(population):
            mask = np.where(np.random.binomial(1,cross_rate,len(p)) == 1)[0]
            crossed = ar(p)
            crossed[[mask]] = sampled_population[i][[mask]]
            new_population.append(crossed)
        return new_population

    def get_good_strings(self,strings,lim=20,unique=False):
        # print strings[0:5]
        fitnesses = [self.fitness(s) for s in strings]
        # print fitnesses[0:10]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        # return sorted_fitnesses
        if unique == False:
            return [strings[i] for i in sorted_fitnesses[0:lim]],[fitnesses[k] for k in sorted_fitnesses[0:lim]]
        else:
            uniques = {}
            good_pop = []
            good_pop_fitnesses = []
            index = 0
            while len(good_pop) < lim and index < len(sorted_fitnesses):
                key = str(strings[sorted_fitnesses[index]])
                if key not in uniques:
                    uniques[key] = 0
                    good_pop.append(strings[sorted_fitnesses[index]])
                    good_pop_fitnesses.append(fitnesses[sorted_fitnesses[index]])
                index += 1
            if len(good_pop) == lim:
                return [good_pop,good_pop_fitnesses]
            else:
                while len(good_pop) < lim:
                    good_pop.append(self.generate_random_string(l=len(strings[0])))
                    good_pop_fitnesses.append(self.fitness(good_pop[-1]))
                return [good_pop,good_pop_fitnesses]
class HIFFProblem(LeftOnes):
    """docstring for HIFF"""
    def __init__(self, corruption_level=0.2,NUMGENES=128,K=2,P=6):
        super(HIFFProblem, self).__init__(corruption_level=corruption_level)
        self.HIFF = HIFF(NUMGENES=128,K=2,P=7)

    @cached_string
    def fitness(self,string,cache=False):
        fitness = self.HIFF.H(string)
        if cache:
            self.cache_fitness(string,fitness)
        return fitness

    def cache_fitness(self,string,fitness):
        self.fitness_cache[self.i_to_s(string)] = fitness

    def i_to_s(self,string):
        return str(list(string))

    def iterative_algorithm(
        self,
        name,
        pop_size=100,
        genome_length=20,
        lim_percentage=20,
        lim=20,
        trials=10,
        corruption_level=0.2,
        num_epochs=50,
        lr = 0.1,
        online_training=False,
        pickle_data=False,
        max_evaluations=200000,
        save_data=False,
        cross_rate=0.9):
        self.mask = np.random.binomial(1,0.5,genome_length)
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA(n_visible=genome_length,n_hidden=[60])
        self.dA.build_dA(corruption_level)
        self.build_sample_dA_2(k=5)
        all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,population_limit)
        self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr,output_folder=results_path,iteration=0)
        # sampled_population = [self.sample_dA([i]) for i in self.get_new_population_rw(all_strings)]
        fit_p_pop = self.get_new_population_rw(all_strings)
        sampled_population = np.array(self.sample_dA(fit_p_pop[0:pop_size/5]),"b").reshape(-1,genome_length)
        print "len sampled:",sampled_population.shape
        new_population = self.create_population_with_unif_cross(sampled_population,fit_p_pop,cross_rate)
        # print "s:",sampled_population
        original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,new_population)
        if save_data:
            self.save_population(sampled_population,0)
            self.save_training_data(good_strings,0)
            random_pop = [self.generate_random_string(genome_length) for z in range(1000)]
            print random_pop[0]
            sampled_r_pop = self.sample_dA(random_pop)
            self.save_sampled_random_population([r for r in sampled_r_pop],0)
            self.save_random_population(random_pop,0)
            self.save_pop_fitnesses(sample_fitnesses,"fitness_pop",0)
            self.save_pop_fitnesses(self.fitness_many(good_strings),"fitness_training_data",0)
        print "writing..."
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses),np.std(original_fitnesses)))
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
        print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        print "writing over"
        for iteration in range(0,trials):
            print "iteration:",iteration
            print "choosing pop:"
            population = self.get_new_population_rw(new_population,sample_fitnesses)
            population[0] = new_population[np.argmax(sample_fitnesses)]
            print "choosing pop over"
            if online_training == False:
                print "building model..."
                self.dA = Deep_dA(n_visible=genome_length,n_hidden=100)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=True)
            # return self.get_good_strings(population,population_limit,unique=True)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr,output_folder=results_path,iteration=iteration+1)
            print "sampling..."
            sampled_population = np.array(self.sample_dA(population[0:pop_size/5]),"b").reshape(-1,genome_length)
            print "sampled pop:",sampled_population.shape
            new_population = self.create_population_with_unif_cross(sampled_population,population,cross_rate)
            new_population[0:1] = good_strings[0:1]
            print "sampling over"
            print "getting_statistics..."
            original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,new_population)
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(sample_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(sample_fitnesses)])
            print "np.argmax(sample_fitnesses):",np.argmax(sample_fitnesses)
            print "new_population[np.argmax(sample_fitnesses)]:",new_population[np.argmax(sample_fitnesses)]
            print "best fitnesses:",sorted(sample_fitnesses,reverse=True)[0:10]
            print new_population[np.argmax(sample_fitnesses)]
            print self.fitness(new_population[np.argmax(sample_fitnesses)])
            print "sample_fitnesses[np.argmax(sample_fitnesses)]:",sample_fitnesses[np.argmax(sample_fitnesses)]
            if save_data:
                print "saving stuff..."
                self.save_population(sampled_population,iteration+1)
                self.save_training_data(good_strings,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(1000)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population([r for r in sampled_r_pop],iteration+1)
                self.save_random_population(random_pop,iteration+1)
                self.save_pop_fitnesses(sample_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()

    def create_population_with_unif_cross(self,sampled_population,population,cross_rate=0.1):
        new_population = []
        for i,p in enumerate(population):
            mask = np.where(np.random.binomial(1,cross_rate,len(p)) == 1)[0]
            crossed = ar(p)
            crossed[[mask]] = sampled_population[i][[mask]]
            new_population.append(crossed)
        return new_population

    def get_good_strings(self,strings,lim=20,unique=False):
        # print strings[0:5]
        fitnesses = [self.fitness(s) for s in strings]
        # print fitnesses[0:10]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        # return sorted_fitnesses
        if unique == False:
            return [strings[i] for i in sorted_fitnesses[0:lim]],[fitnesses[k] for k in sorted_fitnesses[0:lim]]
        else:
            uniques = {}
            good_pop = []
            good_pop_fitnesses = []
            index = 0
            while len(good_pop) < lim and index < len(sorted_fitnesses):
                key = str(strings[sorted_fitnesses[index]])
                if key not in uniques:
                    uniques[key] = 0
                    good_pop.append(strings[sorted_fitnesses[index]])
                    good_pop_fitnesses.append(fitnesses[sorted_fitnesses[index]])
                index += 1
            if len(good_pop) == lim:
                return [good_pop,good_pop_fitnesses]
            else:
                while len(good_pop) < lim:
                    good_pop.append(self.generate_random_string(l=len(strings[0])))
                    good_pop_fitnesses.append(self.fitness(good_pop[-1]))
                return [good_pop,good_pop_fitnesses]
def trap(individual):
    u = sum(individual)
    k = len(individual)
    if u == k:
        return k
    else:
        return k - 1 - u

def inv_trap(individual):
    u = sum(individual)
    k = len(individual)
    if u == 0:
        return k
    else:
        return u - 1

def dec3_trap(individual):
    u = sum(individual)
    if u == 0:
        return 0.9
    if u == 1:
        return 0.8
    if u == 2:
        return 0
    if u == 3:
        return 1

class CF1(HIFFProblem):
    """docstring for HIFF"""
    def __init__(self, corruption_level=0.2,NUMGENES=128,K=2,P=6):
        super(HIFFProblem, self).__init__(corruption_level=corruption_level)
        print self.max_zeros([1]*128)
        self.rr_mask = np.array(np.random.binomial(1,0.5,64),"b")
        # self.MAXSAT = MAXSAT()
        self.fitness_cache = {}

    @cached_string
    def fitness(self,string,cache=False):
        fitness = self.dec_3(string)
        if cache:
            self.cache_fitness(string,fitness)
        return fitness

    def chuang_f1(self,individual):
        """Binary deceptive function from : Multivariate Multi-Model Approach for
        Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.
        
        The function takes individual of 40+1 dimensions and has two global optima
        in [1,1,...,1] and [0,0,...,0].
        """    
        total = 0
        if individual[-1] == 0:
            for i in xrange(0,len(individual)-1,4):
                total += inv_trap(individual[i:i+4])
        else:
            for i in xrange(0,len(individual)-1,4):
                total += trap(individual[i:i+4])
        return total

    def chuang_f2(self,individual):
        """Binary deceptive function from : Multivariate Multi-Model Approach for
        Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.
        
        The function takes individual of 40+1 dimensions and has four global optima
        in [1,1,...,0,0], [0,0,...,1,1], [1,1,...,1] and [0,0,...,0].    
        """    
        total = 0
        if individual[-2] == 0 and individual[-1] == 0:
            for i in xrange(0,len(individual)-2,8):
                total += inv_trap(individual[i:i+4]) + inv_trap(individual[i+4:i+8])
        elif individual[-2] == 0 and individual[-1] == 1:
            for i in xrange(0,len(individual)-2,8):
                total += inv_trap(individual[i:i+4]) + trap(individual[i+4:i+8])
        elif individual[-2] == 1 and individual[-1] == 0:
            for i in xrange(0,len(individual)-2,8):
                total += trap(individual[i:i+4]) + inv_trap(individual[i+4:i+8])
        else:
            for i in xrange(0,len(individual)-2,8):
                total += trap(individual[i:i+4]) + trap(individual[i+4:i+8])
        return total

    def chuang_f3(self,individual):
        """Binary deceptive function from : Multivariate Multi-Model Approach for
        Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.

        The function takes individual of 40+1 dimensions and has two global optima
        in [1,1,...,1] and [0,0,...,0].
        """
        total = 0
        if individual[-1] == 0:
            for i in xrange(0,len(individual)-1,4):
                total += inv_trap(individual[i:i+4])
        else:
            for i in xrange(2,len(individual)-3,4):
                total += inv_trap(individual[i:i+4])
            total += trap(individual[-2:]+individual[:2])
        return total

    # def royal_road1(self,_individual, order=8):
    #     """Royal Road Function R1 as presented by Melanie Mitchell in : 
    #     "An introduction to Genetic Algorithms".
    #     """
    #     individual = _individual^np.array(np.ones(128),"b")
    #     nelem = len(individual) / order
    #     max_value = int(2**order - 1)
    #     total = 0
    #     for i in xrange(nelem):
    #         value = int("".join(map(str, individual[i*order:i*order+order])), 2)
    #         total += int(order) * int(value/max_value)
    #     return total

    def royal_road1(self,_individual, order=8):
        """Royal Road Function R1 as presented by Melanie Mitchell in : 
        "An introduction to Genetic Algorithms".
        """
        individual = _individual^self.rr_mask
        nelem = len(individual) / order
        max_value = int(2**order - 1)
        total = 0
        for i in xrange(nelem):
            value = int("".join(map(str, individual[i*order:i*order+order])), 2)
            total += int(order) * int(value/max_value)
        return total

    def max_zeros(self,individual):
        return len(individual) - sum(individual)

    def max_sat(self,individual,cache=False):
        """Royal Road Function R1 as presented by Melanie Mitchell in : 
        "An introduction to Genetic Algorithms".
        """
        fitness = self.MAXSAT.compFit(individual)
        if cache:
            self.cache_fitness(individual,fitness)
        return fitness
        # return total

    def trap_5(self,_individual,cache=False):
        individual = _individual^self.mask
        fitness = 0
        for i in range(0,len(individual),5):
            fitness += trap(individual[i:i+5])
        if cache:
            self.cache_fitness(_individual,fitness)
        return fitness

    def dec_3(self,_individual,cache=False):
        individual = _individual^self.mask
        fitness = 0
        for i in range(0,len(individual),3):
            fitness += dec3_trap(individual[i:i+3])
        if cache:
            self.cache_fitness(_individual,fitness)
        return fitness


if __name__ == '__main__':
    # args = sys.argv
    # pop_size=int(args[1])
    # print pop_size
    # genome_length=500
    # lim_percentage=int(args[2])
    # lim=int(args[3])
    # trials=10
    # num_epochs=int(args[4])
    # lr = float(args[5])
    # online_training=int(args[6])
    # if online_training == 0:
    #     online_training = False
    # else:
    #     online_training = True
    # pickle_data=False
    # corruption_level=float(args[7])
    # cross_rate=float(args[8])
    # for i in range(0,10):
    #         name = "royal_road_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(pop_size,lim_percentage,num_epochs,lr,online_training,corruption_level,cross_rate,i)
    #         l = HIFFProblem(corruption_level=0.9)
    #         # name= "hiff_test"
    #         z=l.iterative_algorithm(name,
    #                             pop_size=pop_size,
    #                             genome_length=128,
    #                             lim_percentage=lim_percentage,
    #                             lim=20,
    #                             trials=1,
    #                             corruption_level=corruption_level,
    #                             num_epochs=num_epochs,
    #                             lr = lr,
    #                             online_training=online_training,
    #                             pickle_data=False,
    #                             save_data=False,
    #                             max_evaluations=500000,
    #                             cross_rate=cross_rate)



    # l = LeftOnes(corruption_level=0.1)
    # # l=ACSKnapsack(corruption_level=0.2,knapsack_file="weing8.pkl")
    # name= "l1_test"
    # l.iterative_algorithm(name,
    #                     pop_size=1000,
    #                     genome_length=100,
    #                     lim_percentage=0,
    #                     lim=20,
    #                     trials=1,
    #                     corruption_level=0.1,
    #                     num_epochs=100,
    #                     lr = 0.1,
    #                     online_training=False,
    #                     pickle_data=False,
    #                     save_data=True,
    #                     max_evaluations=10000)
    # l=ACSKnapsack(corruption_level=0.1,knapsack_file="knapsack_500.pkl")
    # name= "l1_test"
    # l.iterative_algorithm(name,
    #                     pop_size=10000,
    #                     genome_length=500,
    #                     lim_percentage=10,
    #                     lim=20,
    #                     trials=1,
    #                     corruption_level=0.01,
    #                     num_epochs=100,
    #                     lr = 0.01,
    #                     online_training=True,
    #                     pickle_data=False,
    #                     save_data=True,
    #                     max_evaluations=200000)

    # l = CF1(corruption_level=0.9)
    # l=ACSKnapsack(corruption_level=0.9,knapsack_file="weing8.pkl")
    # for i in range(0,10):
    #     name= "knapsack_{0}".format(i)
    #     z=l.iterative_algorithm(name,
    #                         pop_size=5000,
    #                         genome_length=105,
    #                         lim_percentage=10,
    #                         lim=10,
    #                         trials=1,
    #                         corruption_level=0.05,
    #                         num_epochs=50,
    #                         lr = 0.01,
    #                         online_training=True,
    #                         pickle_data=False,
    #                         save_data=False,
    #                         max_evaluations=500000)
    # for i in range(0,10):
    #     name = "low-corruption_maxsat_{0}".format(i)
    #     z=l.iterative_algorithm(name,
    #                         pop_size=5000,
    #                         genome_length=100,
    #                         lim_percentage=10,
    #                         lim=20,
    #                         trials=1,
    #                         corruption_level=0.05,
    #                         num_epochs=50,
    #                         lr = 0.1,
    #                         online_training=True,
    #                         pickle_data=False,
    #                         save_data=False,
    #                         max_evaluations=500000)
    l=CF1()
    name= "hiff_test___"
    z=l.iterative_algorithm(name,
                        pop_size=5000,
                        genome_length=60,
                        lim_percentage=10,
                        lim=20,
                        trials=1,
                        corruption_level=0.05,
                        num_epochs=25,
                        lr = 0.1,
                        online_training=True,
                        pickle_data=False,
                        save_data=False,
                        max_evaluations=500000,
                        cross_rate=0.9)


