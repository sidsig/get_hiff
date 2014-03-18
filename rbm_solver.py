from left_ones import *
from scipy.spatial.distance import pdist,cdist
import pdb
from deep_da import Deep_dA_cont
import sklearn.cluster
import pdb

DO_CACHING = False
FITNESS_F = "sphere"
# def cached_string(f):
#     """
#     This is a decorator to make sure
#     the should still be active
#     """
#     def wrapped(self,*args):
#         string = self.i_to_s(args[0])
#         if string in self.fitness_cache:
#             return self.fitness_cache[string]
#         else:
#             return f(self,*args,cache=True)
#     return wrapped

class RBMSolver(CF1):
    """docstring for RBMSolver"""
    def __init__(self,knapsack_file="knapsack_500.pkl"):
        super(RBMSolver, self).__init__(corruption_level=0.0)
        self.rr_mask = np.array(np.random.binomial(1,0.5,64),"b")
        self.RBM = RBM(n_visible=64,n_hidden=50)
        self.knapsack = pickle.load(open(knapsack_file))
        self.fitness_cache = {}
        self.HIFF = HIFF(NUMGENES=128,K=2,P=7)
        self.population = []
        self.population_fitnesses = []
        self.sample_fitnesses = []
    # def fitness(self,string):
    #     return self.royal_road1(string)
    # @cached_string
    if FITNESS_F == "knapsack_105":
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
    elif FITNESS_F == "hiff":
        def fitness(self,string,cache=False):
            fitness = self.HIFF.H(string)
            if cache:
                self.cache_fitness(fitness)
            return fitness
    elif FITNESS_F == "max_sat":
        @cached_string
        def fitness(self,string,cache=False):
            return self.max_sat(string,cache)
    elif FITNESS_F == "dec_3":
        def fitness(self,string,cache=False):
            fitness = self.dec_3(string)
            if cache:
                # print "caching"
                self.cache_fitness(fitness)
            return fitness
    elif FITNESS_F == "max_ones":
        def fitness(self,string,cache=False):
            fitness = np.sum(string^self.mask)
            if cache:
                self.cache_fitness(fitness)
            return fitness
    elif FITNESS_F == "left_ones":
        def fitness(self,_string,cache=False):
            string =_string^self.mask
            fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
            if cache:
                self.cache_fitness(fitness)
            return fitness
    elif FITNESS_F == "sphere":
        def fitness(self,string,cache=False):
            genome = np.array(string)
            genome = (genome*2)-1
            fitness = np.sum(genome**2)*-1
            if cache:
                self.cache_fitness(fitness)
            return fitness
    elif FITNESS_F == "rosenbrock":
        def fitness(self,string,cache=False):
            genome = np.array(string)
            genome = (genome*2.048*2)-2.048
            fitness = sum(100 * (x * x - y)**2 + (1. - x)**2 \
                   for x, y in zip(genome[:-1], genome[1:]))
            fitness = fitness * -1
            if cache:
                self.cache_fitness(fitness)
            return fitness
    elif FITNESS_F == "rastrigin":
        def fitness(self,string,cache=False):
            genome = np.array(string)
            genome = (genome*10.24)-5.12
            fitness = 10 * len(string) + sum(gene * gene - 10 * \
                        np.cos(2 * np.pi * gene) for gene in string)
            fitness = fitness * -1
            if cache:
                self.cache_fitness(fitness)
            return fitness

    def cache_fitness(self,string,fitness):
        # print "caching"
        self.fitness_cache[self.i_to_s(string)] = fitness

    def tournament_selection_replacement(self,population, fitnesses=None, pop_size=None):
        if pop_size == None:
            pop_size = len(population)
        if fitnesses == None:
            fitnesses = self.fitness_many(population)
        new_population = []
        while len(new_population) < pop_size:
            child_1 = int(np.random.random() * pop_size)
            child_2 = int(np.random.random() * pop_size)
            if fitnesses[child_1] > fitnesses[child_2]:
                new_population.append(copy.deepcopy(population[child_1]))
            else:
                new_population.append(copy.deepcopy(population[child_2]))
        return new_population

    def train_RBM(self,data,num_epochs=200,lr=0.1,output_folder="",):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.RBM.params,[self.RBM.input],self.RBM.cost,train_set,num_epochs=num_epochs,consider_constant=[self.RBM.consider_constant],updates_old=self.RBM.updates,save=False)
    def build_sample_RBM(self,):
        self.sample_RBM = theano.function([self.RBM.input],self.RBM.v_sample,updates=self.RBM.updates)

    def build_sample_RBM_2(self,k=20):
        samples,updates = self.RBM.build_sampler(k=k)
        self.sample_RBM = theano.function([self.RBM.input],samples,updates=updates)

    # def iterative_algorithm(
    #     self,
    #     name,
    #     pop_size=100,
    #     genome_length=20,
    #     lim_percentage=20,
    #     lim=20,
    #     trials=10,
    #     corruption_level=0.2,
    #     num_epochs=50,
    #     lr = 0.1,
    #     online_training=False,
    #     pickle_data=False,
    #     max_evaluations=200000,
    #     save_data=False,
    #     cross_rate=0.9,
    #     unique_training=False,
    #     sample_rate=5,
    #     hiddens=40):
    #     results_path = "results/autoencoder/{0}/".format(name)
    #     ensure_dir(results_path)
    #     trials = max_evaluations/pop_size
    #     population_limit = lim
    #     cross_rate = cross_rate
    #     if lim_percentage > 0:
    #         population_limit = int(pop_size*(lim_percentage/100.0))
    #         print "population_limit:",population_limit
    #         print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
    #     fitfile = open("{0}fitnesses.dat".format(results_path),"w")
    #     self.RBM = RBM(n_visible=genome_length,n_hidden=20)
    #     self.RBM.build_RBM(k=2)
    #     self.build_sample_RBM()
    #     all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,population_limit)
    #     self.train_RBM(ar(good_strings),num_epochs=num_epochs,lr=lr,output_folder=results_path)
    #     fit_p_pop = self.get_new_population_rw(all_strings)
    #     sampled_population = self.sample_RBM(fit_p_pop)
    #     new_population = self.create_population_with_unif_cross(sampled_population,fit_p_pop,cross_rate)
    #     original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,new_population)
    #     fitfile.write("{0},{1},{2},{3}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses),np.std(original_fitnesses)))
    #     fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
    #     print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
    #     print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
    #     for iteration in range(0,trials):
    #         print "iteration:",iteration
    #         population = self.get_new_population_rw(new_population,sample_fitnesses)
    #         population[0] = new_population[np.argmax(sample_fitnesses)]
    #         if online_training == False:
    #             print "building model..."
    #             self.dA = dA(n_visible=genome_length,n_hidden=100)
    #             self.dA.build_dA(corruption_level)
    #             self.build_sample_dA()
    #         good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=True)
    #         # return self.get_good_strings(population,population_limit,unique=True)
    #         for f in good_strings_fitnesses:
    #             print "good_strings_fitnesses:",f
    #         if iteration % 10 == 0:
    #             self.RBM.build_RBM(k=1)
    #         self.train_RBM(ar(good_strings),num_epochs=num_epochs,lr=lr,output_folder=results_path)
    #         print "sampling..."
    #         sampled_population = self.sample_RBM(population)
    #         new_population = self.create_population_with_unif_cross(sampled_population,population,cross_rate)
    #         new_population[0:1] = good_strings[0:1]
    #         print "sampling over"
    #         print "getting_statistics..."
    #         original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,new_population)
    #         print "statistics over"
    #         fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(sample_fitnesses)))
    #         fitfile.flush()
    #         print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
    #         print "best from previous:",self.fitness(new_population[np.argmax(sample_fitnesses)])
    #     fitfile.close()

    def get_good_strings(self,strings,lim=20,unique=False,fitnesses=None):
        # print strings[0:5]
        if fitnesses == None:
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

    def hboa_iterative_algorithm(
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
        cross_rate=0.9,
        unique_training=False,
        sample_rate=5,
        hiddens=40
        ):
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
        # sample_rate = pop_size/population_limit
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.RBM = RBM(n_visible=genome_length,n_hidden=hiddens)
        self.RBM.build_RBM(k=1)
        self.build_sample_RBM_2(k=sample_rate)
        all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,population_limit)
        self.train_RBM(ar(good_strings),num_epochs=num_epochs,lr=lr,output_folder=results_path)
        fit_p_pop = self.get_new_population_rw(all_strings)
        sampled_population = np.array(self.sample_RBM(fit_p_pop),"b").reshape(-1,genome_length)
        # pdb.set_trace()
        print "len(sampled_population):",len(sampled_population)
        self.sample_fitnesses = self.fitness_many(sampled_population)
        new_population = self.RTR(all_strings,sampled_population,self.fitness_many(all_strings),self.sample_fitnesses,w=100)
        original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,new_population)
        self.population_fitnesses = sample_fitnesses
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses),np.std(original_fitnesses)))
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
        print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        for iteration in range(0,trials):
            try:
                print "iteration:",iteration
                population = new_population
                self.population = new_population
                # rw = self.get_new_population_rw(population)
                rw = self.tournament_selection_replacement(population)
                # population[0] = new_population[np.argmax(sample_fitnesses)]
                if online_training == False:
                    print "building model..."
                    self.dA = dA(n_visible=genome_length,n_hidden=100)
                    self.dA.build_dA(corruption_level)
                    self.build_sample_dA()
                # if iteration % 50 == 0:
                print "building rbm..."
                # self.RBM = RBM(n_visible=genome_length,n_hidden=hiddens)
                # self.RBM.build_RBM(k=1)
                # self.build_sample_RBM_2(k=sample_rate)
                good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=unique_training,fitnesses=self.population_fitnesses)
                # return self.get_good_strings(population,population_limit,unique=True)
                for f in good_strings_fitnesses:
                    print "good_strings_fitnesses:",f
                print "training rbm"
                # self.train_RBM(ar(good_strings),num_epochs=num_epochs,lr=lr,output_folder=results_path)
                self.train_RBM(ar(rw)[0:population_limit],num_epochs=num_epochs,lr=lr,output_folder=results_path)
                print "training rbm over"
                print "sampling..."
                sampled_population = np.array(self.sample_RBM(rw),"b").reshape(-1,genome_length)
                print "len(sampled_population):",len(sampled_population)
                self.sample_fitnesses = self.fitness_many(sampled_population)
                new_population = self.RTR(population,sampled_population,population_fitnesses=self.population_fitnesses,sample_fitnesses=self.sample_fitnesses,w=100)
                print "sampling over"
                print "getting_statistics..."
                # original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,new_population)
                print "statistics over"
                fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
                fitfile.flush()
                print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
                print "best from previous:",self.fitness(new_population[np.argmax(self.population_fitnesses)])
            except KeyboardInterrupt:
                return new_population
        fitfile.close()
        return new_population

    # def RTR(self,population,sampled_population,w=None):
    #     if w == None:
    #         w = len(population)/20
    #     _population = np.array(population)
    #     for individual in sampled_population:
    #         indexes = np.random.choice(len(population), w, replace=False)
    #         distances = [distance.hamming(individual,population[i]) for i in indexes[0:w]]
    #         replacement = indexes[np.argmin(distances)]
    #         if self.fitness_cache[self.i_to_s(population[replacement])] < self.fitness_cache[self.i_to_s(individual)]:
    #             population[replacement] = individual
    #     return population

    # def RTR(self,population,sampled_population,population_fitnesses,sample_fitnesses,w=None):
    #     print "start RTR"
    #     if w == None:
    #         w = len(population)/20
    #     _population = np.array(population)
    #     # print "w:",w
    #     for ind_i,individual in enumerate(sampled_population):
    #         # print "individual:",individual
    #         print "ind_i:",ind_i
    #         indexes = np.random.choice(len(_population), w, replace=False)
    #         # print "indexes:",indexes
    #         distances = pdist(np.vstack((individual,_population[indexes])))[0:w]
    #         # print "distances:",distances
    #         replacement = indexes[np.argmin(distances)]
    #         # print "min:",np.argmin(distances)
    #         # print "replacement:",replacement
    #         # print "fitness _population[replacement]):",self.fitness_cache[self.i_to_s(_population[replacement])]
    #         # print "fitness individual:",self.fitness_cache[self.i_to_s(individual)]
    #         if DO_CACHING:
    #             if self.fitness_cache[self.i_to_s(_population[replacement])] < self.fitness_cache[self.i_to_s(individual)]:
    #                 _population[replacement] = individual
    #         else:
    #             if population_fitnesses[replacement] < sample_fitnesses[ind_i]:
    #                 _population[replacement] = individual
    #                 population_fitnesses[replacement] = sample_fitnesses[ind_i]
    #     print "end RTR"
    #     return _population

    def RTR(self,population,sampled_population,population_fitnesses,sample_fitnesses,w=None):
        print "start RTR"
        if w == None:
            w = len(population)/20
        _population = np.array(population)
        # print "w:",w
        for ind_i,individual in enumerate(sampled_population):
            # print "individual:",individual
            # print "ind_i:",ind_i
            indexes = np.random.choice(len(_population), w, replace=False)
            # print "indexes:",indexes
            distances = cdist(_population[indexes],[individual],"hamming")
            # print "distances:",distances
            replacement = indexes[np.argmin(distances.flatten())]
            # print "min:",np.argmin(distances)
            # print "replacement:",replacement
            # print "fitness _population[replacement]):",self.fitness_cache[self.i_to_s(_population[replacement])]
            # print "fitness individual:",self.fitness_cache[self.i_to_s(individual)]
            if population_fitnesses[replacement] < sample_fitnesses[ind_i]:
                _population[replacement] = individual
                population_fitnesses[replacement] = sample_fitnesses[ind_i]
        print "end RTR"
        return _population

    def i_to_s(self,string):
        return str(list(string))

class AESolver(RBMSolver):
    """docstring for RBMSolver"""
    def __init__(self,knapsack_file="knapsack_500.pkl"):
        super(AESolver, self).__init__(knapsack_file="knapsack_500.pkl")
        self.rr_mask = np.array(np.random.binomial(1,0.5,64),"b")
        self.RBM = RBM(n_visible=64,n_hidden=50)
        self.knapsack = pickle.load(open(knapsack_file))
        self.fitness_cache = {}

    def save_population(self,name,population,iteration):
        np.savetxt("results/{0}/population_{1}.dat".format(name,iteration),population)

    def save_sampled_rw_population(self,name,population,iteration):
        np.savetxt("results/{0}/rw_population_{1}.dat".format(name,iteration),population)

    def save_training_data(self,name,tdata,iteration):
        np.savetxt("results/{0}/training_data_{1}.dat".format(name,iteration),tdata)

    def save_sampled_random_population(self,name,population,iteration):
        np.savetxt("results/{0}/random_sampled_population_{1}.dat".format(name,iteration),population)

    def save_pop_fitnesses(self,name,fitnesses,fname,iteration):
        np.savetxt("results/{0}/{1}_{2}.dat".format(name,fname,iteration),fitnesses)

    def save_random_population(self,name,population,iteration):
        np.savetxt("results/{0}/random_population_{1}.dat".format(name,iteration),population)

    def hboa_iterative_algorithm(
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
        cross_rate=0.9,
        unique_training=False,
        sample_rate=5,
        hiddens=40,
        use_best_strings=True,
        rtr = True
        ):
        self.mask = np.random.binomial(1,0.5,genome_length)
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        ensure_dir("results/{0}/".format(name))
        np.savetxt("results/{0}/mask.dat".format(name),self.mask)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = dA(n_visible=genome_length,n_hidden=hiddens)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        new_population = np.random.binomial(1,0.5,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
        if save_data:
            print "saving stuff..."
            self.save_population(name,new_population,0)
            self.save_sampled_rw_population(name,sampled_population,0)
            self.save_training_data(name,training_data,0)
            random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
            print random_pop[0]
            sampled_r_pop = self.sample_dA(random_pop)
            self.save_sampled_random_population(name,[r for r in sampled_r_pop],0)
            self.save_random_population(name,random_pop,0)
            self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",0)
            self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",0)
            print "saving over"
        for iteration in range(0,trials):
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            rw = self.tournament_selection_replacement(population)
            if online_training == False:
                print "building model..."
                self.dA = dA(n_visible=genome_length,n_hidden=100)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=unique_training,fitnesses=self.population_fitnesses)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            print "training A/E"
            if use_best_strings:
                training_data = np.array(good_strings)
            else:
                training_data = rw[0:population_limit]
            self.train_dA(training_data,num_epochs=num_epochs,lr=lr,output_folder=results_path)
            print "training A/E over"
            print "sampling..."
            sampled_population = np.array(self.sample_dA(rw),"b")
            print "len(sampled_population):",len(sampled_population)
            self.sample_fitnesses = self.fitness_many(sampled_population)
            if rtr:
                new_population = self.RTR(population,sampled_population,population_fitnesses=self.population_fitnesses,sample_fitnesses=self.sample_fitnesses,w=pop_size/10)
            else:
                new_population = sampled_population
            print "sampling over"
            print "getting_statistics..."
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(self.population_fitnesses)])
            if save_data:
                print "saving stuff..."
                self.save_population(name,new_population,iteration+1)
                self.save_sampled_rw_population(name,sampled_population,iteration+1)
                self.save_training_data(name,training_data,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population(name,[r for r in sampled_r_pop],iteration+1)
                self.save_random_population(name,random_pop,iteration+1)
                self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()
        return new_population

class AEContinuousSolver(AESolver):
    """docstring for RBMSolver"""
    def __init__(self,knapsack_file="knapsack_500.pkl"):
        super(AESolver, self).__init__(knapsack_file="knapsack_500.pkl")
        self.rr_mask = np.array(np.random.binomial(1,0.5,64),"b")
        self.RBM = RBM(n_visible=64,n_hidden=50)
        self.knapsack = pickle.load(open(knapsack_file))
        self.fitness_cache = {}
        self.sample_sd = 0.01

    def build_sample_dA(self):  
        self.sample_dA = theano.function([self.dA.input],self.dA.sample)

    def sample_dA_continuous(self,data):
        means_matrix = self.sample_dA(data)
        np.savetxt("mm.dat",means_matrix)
        samples = np.array([np.random.multivariate_normal(means_vector,self.cov) for means_vector in means_matrix])
        # bounds
        samples[np.where(samples>1)]=1
        samples[np.where(samples<0)]=0
        return samples

    def sample_dA_chrisantha(self,data):
        means_matrix = self.sample_dA(data)
        pop_size = len(means_matrix)
        np.savetxt("mm.dat",means_matrix)
        means_matrix = np.mean(means_matrix,axis=0)
        samples = np.array([np.random.multivariate_normal(means_matrix,self.cov) for i in range(pop_size)])
        # bounds
        samples[np.where(samples>1)]=1
        samples[np.where(samples<0)]=0
        return samples

    def sample_dA_not_probabilistic(self,_data,sd=0.00):
        if sd > 0:
            noise = np.array([np.random.multivariate_normal([0]*len(_data[0]),np.eye(len(_data[0]))*sd) for i in range(len(_data))])
            data = _data + noise
        else:
            data = _data
        samples = self.sample_dA(data)
        samples[np.where(samples>1)]=1
        samples[np.where(samples<0)]=0
        return samples

    def hboa_iterative_algorithm(
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
        cross_rate=0.9,
        unique_training=False,
        sample_rate=5,
        hiddens=40,
        use_best_strings=True,
        w = 50,
        sample_sd = 0.01
        ):
        #pdb.set_trace()
        self.mask = np.random.binomial(1,0.5,genome_length)
        self.sample_sd = sample_sd
        self.cov = np.eye(genome_length)*self.sample_sd
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        ensure_dir("results/{0}/".format(name))
        np.savetxt("results/{0}/mask.dat".format(name),self.mask)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        new_population = np.random.uniform(0.0,1.0,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
        # if save_data:
        #     print "saving stuff..."
        #     self.save_population(name,new_population,0)
        #     self.save_sampled_rw_population(name,sampled_population,0)
        #     self.save_training_data(name,training_data,0)
        #     random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
        #     print random_pop[0]
        #     sampled_r_pop = self.sample_dA(random_pop)
        #     self.save_sampled_random_population(name,[r for r in sampled_r_pop],0)
        #     self.save_random_population(name,random_pop,0)
        #     self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",0)
        #     self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",0)
        #     print "saving over"
        for iteration in range(0,trials):
            # if iteration == 30:
            #     self.sample_sd = 0.001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            # if iteration == 50:
            #     self.sample_sd = 0.0001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            # if iteration == 70:
            #     self.sample_sd = 0.00001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            # if iteration == 80:
            #     self.sample_sd = 0.000001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            print "population:"
            print (np.mean(population,axis=0)*10.24)-5.12
            print (np.std(population,axis=0))
            rw = self.tournament_selection_replacement(population)
            if online_training == False and iteration % 20 == 0:
                print "building model..."
                self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=unique_training,fitnesses=self.population_fitnesses)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            print "training A/E"
            if use_best_strings:
                training_data = np.array(good_strings)
            else:
                training_data = rw[0:population_limit]
            self.train_dA(training_data,num_epochs=num_epochs,lr=lr,output_folder=results_path)
            print "training A/E over"
            print "sampling..."
            sampled_population = np.array(self.sample_dA_continuous(rw))
            print "len(sampled_population):",len(sampled_population)
            self.sample_fitnesses = self.fitness_many(sampled_population)
            new_population = self.RTR(population,sampled_population,population_fitnesses=self.population_fitnesses,sample_fitnesses=self.sample_fitnesses,w=w)
            print "sampling over"
            print "getting_statistics..."
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(self.population_fitnesses)])
            if save_data:
                print "saving stuff..."
                self.save_population(name,new_population,iteration+1)
                self.save_sampled_rw_population(name,sampled_population,iteration+1)
                self.save_training_data(name,training_data,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population(name,[r for r in sampled_r_pop],iteration+1)
                self.save_random_population(name,random_pop,iteration+1)
                self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()
        return new_population

    def chrisantha_algorithm(
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
        cross_rate=0.9,
        unique_training=False,
        sample_rate=5,
        hiddens=40,
        use_best_strings=True,
        w = 20,
        sample_sd = 0.0001
        ):
        self.mask = np.random.binomial(1,0.5,genome_length)
        self.sample_sd = sample_sd
        self.cov = np.eye(genome_length)*self.sample_sd
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        ensure_dir("results/{0}/".format(name))
        np.savetxt("results/{0}/mask.dat".format(name),self.mask)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        new_population = np.random.uniform(0.0,1.0,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
        for iteration in range(0,trials):
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            print "population:"
            print (np.mean(population,axis=0)*5.12*2)-5.12
            print (np.std(population,axis=0))
            rw = self.tournament_selection_replacement(population)
            if online_training == False and iteration % 20 == 0:
                print "building model..."
                self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=unique_training,fitnesses=self.population_fitnesses)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            print "training A/E"
            if use_best_strings:
                training_data = np.array(good_strings)
            else:
                training_data = rw[0:population_limit]
            self.train_dA(training_data,num_epochs=num_epochs,lr=lr,output_folder=results_path)
            print "training A/E over"
            print "sampling..."
            sampled_population = np.array(self.sample_dA_chrisantha(rw))
            print "len(sampled_population):",len(sampled_population)
            self.sample_fitnesses = self.fitness_many(sampled_population)
            new_population = self.RTR(population,sampled_population,population_fitnesses=self.population_fitnesses,sample_fitnesses=self.sample_fitnesses,w=w)
            print "sampling over"
            print "getting_statistics..."
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(self.population_fitnesses)])
            if save_data:
                print "saving stuff..."
                self.save_population(name,new_population,iteration+1)
                self.save_sampled_rw_population(name,sampled_population,iteration+1)
                self.save_training_data(name,training_data,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population(name,[r for r in sampled_r_pop],iteration+1)
                self.save_random_population(name,random_pop,iteration+1)
                self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()
        return new_population

    def not_probabilistic_ia(
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
        cross_rate=0.9,
        unique_training=False,
        sample_rate=5,
        hiddens=40,
        use_best_strings=True,
        w = 20, 
        sample_sd = 0.0001
        ): 
        self.mask = np.random.binomial(1,0.5,genome_length)
        self.sample_sd = sample_sd
        self.cov = np.eye(genome_length)*self.sample_sd
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        ensure_dir("results/{0}/".format(name))
        np.savetxt("results/{0}/mask.dat".format(name),self.mask)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        new_population = np.random.uniform(0.0,1.0,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
        for iteration in range(0,trials):
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            print "population:"
            print (np.mean(population,axis=0)*2.048*2)-2.048
            print (np.std(population,axis=0))
            rw = self.tournament_selection_replacement(population)
            if online_training == False and iteration % 20 == 0:
                print "building model..."
                self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=unique_training,fitnesses=self.population_fitnesses)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            print "training A/E"
            if use_best_strings:
                training_data = np.array(good_strings)
            else:
                training_data = rw[0:population_limit]
            self.train_dA(training_data,num_epochs=num_epochs,lr=lr,output_folder=results_path)
            print "training A/E over"
            print "sampling..."
            sampled_population = np.array(self.sample_dA_not_probabilistic(rw,sd=0.0))
            sampled_population = [self.gradual_movement(parent,proposal) for parent,proposal in zip(rw,sampled_population)]
            print "len(sampled_population):",len(sampled_population)
            self.sample_fitnesses = self.fitness_many(sampled_population)
            new_population = self.RTR(population,sampled_population,population_fitnesses=self.population_fitnesses,sample_fitnesses=self.sample_fitnesses,w=w)
            print "sampling over"
            print "getting_statistics..."
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(self.population_fitnesses)])
            if save_data:
                print "saving stuff..."
                self.save_population(name,new_population,iteration+1)
                self.save_sampled_rw_population(name,sampled_population,iteration+1)
                self.save_training_data(name,training_data,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population(name,[r for r in sampled_r_pop],iteration+1)
                self.save_random_population(name,random_pop,iteration+1)
                self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()
        return new_population

    def RTR(self,population,sampled_population,population_fitnesses,sample_fitnesses,w=None):
        print "start RTR"
        if w == None:
            w = len(population)/20
        _population = np.array(population)
        # print "w:",w
        for ind_i,individual in enumerate(sampled_population):
            # print "individual:",individual
            # print "ind_i:",ind_i
            indexes = np.random.choice(len(_population), w, replace=False)
            # print "indexes:",indexes
            distances = cdist(_population[indexes],[individual],"euclidean")
            # print "distances:",distances
            replacement = indexes[np.argmin(distances.flatten())]
            # print "min:",np.argmin(distances)
            # print "replacement:",replacement
            # print "fitness _population[replacement]):",self.fitness_cache[self.i_to_s(_population[replacement])]
            # print "fitness individual:",self.fitness_cache[self.i_to_s(individual)]
            if population_fitnesses[replacement] < sample_fitnesses[ind_i]:
                _population[replacement] = individual
                population_fitnesses[replacement] = sample_fitnesses[ind_i]
        print "end RTR"
        return _population

    def gradual_movement(self,parent,proposal,lr=0.1,noise=0.001):
        child = (parent*(1-lr))+(proposal*lr)+np.random.normal(0,noise,len(parent))
        return child

    def get_k_means(self,data,n_clusters=20):
        k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
        print 'Finding k centroids using k-means'
        k_means.fit(data)
        predictions = k_means.predict(data)
        cluster_data = dict([(i,[]) for i in xrange(n_clusters)])
        for example,cluster in zip(data,predictions):
            cluster_data[cluster].append(example)
        for cluster in cluster_data.keys():
            cluster_data[cluster].sort(key=lambda x:self.fitness(x))
        centroids = k_means.cluster_centers_    
        return centroids,cluster_data

    def pick_top_k(self,cluster_data,k):
        top_examples = []
        for cluster in cluster_data.keys():
            top_examples.extend(cluster_data[cluster][:k])
        return numpy.array(top_examples)

    def k_means(
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
        cross_rate=0.9,
        unique_training=False,
        sample_rate=5,
        hiddens=40,
        use_best_strings=True,
        w = 50,
        sample_sd = 0.01
        ):
        self.mask = np.random.binomial(1,0.5,genome_length)
        self.sample_sd = sample_sd
        self.cov = np.eye(genome_length)*self.sample_sd
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        ensure_dir("results/{0}/".format(name))
        np.savetxt("results/{0}/mask.dat".format(name),self.mask)
        trials = max_evaluations/pop_size
        population_limit = lim
        cross_rate = cross_rate
        if lim_percentage > 0:
            population_limit = int(pop_size*(lim_percentage/100.0))
            print "population_limit:",population_limit
            print "{0}*({1}/100.0) = {2}".format(pop_size,lim_percentage,int(pop_size*(lim_percentage/100.0)))
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        new_population = np.random.uniform(0.0,1.0,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
        # if save_data:
        #     print "saving stuff..."
        #     self.save_population(name,new_population,0)
        #     self.save_sampled_rw_population(name,sampled_population,0)
        #     self.save_training_data(name,training_data,0)
        #     random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
        #     print random_pop[0]
        #     sampled_r_pop = self.sample_dA(random_pop)
        #     self.save_sampled_random_population(name,[r for r in sampled_r_pop],0)
        #     self.save_random_population(name,random_pop,0)
        #     self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",0)
        #     self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",0)
        #     print "saving over"
        for iteration in range(0,trials):
            # if iteration == 30:
            #     self.sample_sd = 0.001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            # if iteration == 50:
            #     self.sample_sd = 0.0001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            # if iteration == 70:
            #     self.sample_sd = 0.00001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            # if iteration == 80:
            #     self.sample_sd = 0.000001
            #     self.cov = np.eye(genome_length)*self.sample_sd
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            print "population:"
            print (np.mean(population,axis=0)*10.24)-5.12
            print (np.std(population,axis=0))
            rw = self.tournament_selection_replacement(population)
            if online_training == False and iteration % 20 == 0:
                print "building model..."
                self.dA = Deep_dA_cont(n_visible=genome_length,n_hidden=hiddens,squared_err=True)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit,unique=unique_training,fitnesses=self.population_fitnesses)
            print '************'
            centroids,cluster_data = self.get_k_means(population)
            for f in good_strings_fitnesses:
                print "good_strings_fitnesses:",f
            print "training A/E"
            if use_best_strings:
                training_data = np.array(good_strings)
            else:
                #training_data = centroids
                training_data = self.pick_top_k(cluster_data,1)
            self.train_dA(training_data,num_epochs=num_epochs,lr=lr,output_folder=results_path)
            print "training A/E over"
            print "sampling..."
            sampled_population = np.array(self.sample_dA_continuous(rw))
            print "len(sampled_population):",len(sampled_population)
            self.sample_fitnesses = self.fitness_many(sampled_population)
            new_population = self.RTR(population,sampled_population,population_fitnesses=self.population_fitnesses,sample_fitnesses=self.sample_fitnesses,w=w)
            print "sampling over"
            print "getting_statistics..."
            print "statistics over"
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses))
            print "best from previous:",self.fitness(new_population[np.argmax(self.population_fitnesses)])
            if save_data:
                print "saving stuff..."
                self.save_population(name,new_population,iteration+1)
                self.save_sampled_rw_population(name,sampled_population,iteration+1)
                self.save_training_data(name,training_data,iteration+1)
                random_pop = [self.generate_random_string(genome_length) for z in range(pop_size)]
                print random_pop[0]
                sampled_r_pop = self.sample_dA(random_pop)
                self.save_sampled_random_population(name,[r for r in sampled_r_pop],iteration+1)
                self.save_random_population(name,random_pop,iteration+1)
                self.save_pop_fitnesses(name,self.population_fitnesses,"fitness_pop",iteration+1)
                self.save_pop_fitnesses(name,good_strings_fitnesses,"fitness_training_data",iteration+1)
                print "saving over"
        fitfile.close()
        return new_population


if __name__ == '__main__':
    # args = sys.argv
    # pop_size=int(args[1])
    # print pop_size
    # genome_length=64
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
    # unique_training=int(args[7])
    # if unique_training == 0:
    #     unique_training = False
    # else:
    #     unique_training = True
    # pickle_data=False
    # sample_rate=int(args[8])
    # hiddens=int(args[9])
    # corruption_level=int(args[10])
    # for i in range(0,5):
    #     name = "hiff_64_online_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}".format(pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,hiddens,corruption_level,i)
    #     l = RBMSolver()
    #     z=l.hboa_iterative_algorithm(name,
    #                         pop_size=pop_size,
    #                         genome_length=genome_length,
    #                         lim_percentage=lim_percentage,
    #                         lim=20,
    #                         trials=1,
    #                         corruption_level=corruption_level,
    #                         num_epochs=num_epochs,
    #                         lr = lr,
    #                         online_training=True,
    #                         pickle_data=False,
    #                         save_data=False,
    #                         max_evaluations=500000,
    #                         cross_rate=1.0,
    #                         unique_training=unique_training,
    #                         sample_rate=sample_rate,
    #                         hiddens=hiddens)
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
    # unique_training=int(args[7])
    # if unique_training == 0:
    #     unique_training = False
    # else:
    #     unique_training = True
    # pickle_data=False
    # sample_rate=int(args[8])
    # hiddens=int(args[9])
    # corruption_level=float(args[10])
    # use_best_strings=int(args[11])
    # if use_best_strings == 0:
    #     use_best_strings = False
    # else:
    #     use_best_strings = True
    # trial = int(args[12])
    # name = "ae_knapsack_500_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}".format(pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,hiddens,corruption_level,use_best_strings,trial)
    # l = AESolver()
    # z=l.hboa_iterative_algorithm(name,
    #                     pop_size=pop_size,
    #                     genome_length=genome_length,
    #                     lim_percentage=lim_percentage,
    #                     lim=20,
    #                     trials=1,
    #                     corruption_level=corruption_level,
    #                     num_epochs=num_epochs,
    #                     lr = lr,
    #                     online_training=online_training,
    #                     pickle_data=False,
    #                     save_data=False,
    #                     max_evaluations=200000,
    #                     cross_rate=0.9,
    #                     unique_training=unique_training,
    #                     sample_rate=sample_rate,
    #                     hiddens=hiddens,
    #                     use_best_strings=use_best_strings)

    # for i in range(0,1):
    #     corruption_level = 0.05
    #     name = "ae_knapsack_500_p600_10_50_0.1_True_True_0_200_True_{0}".format(corruption_level)
    #     l = AESolver()
    #     z=l.hboa_iterative_algorithm(name,
    #                         pop_size=5000,
    #                         genome_length=128,
    #                         lim_percentage=10,
    #                         lim=20,
    #                         trials=1,
    #                         corruption_level=corruption_level,
    #                         num_epochs=25,
    #                         lr = 0.1,
    #                         online_training=True,
    #                         pickle_data=False,
    #                         save_data=False,
    #                         max_evaluations=500000,
    #                         cross_rate=1.0,
    #                         unique_training=True,
    #                         sample_rate=1,
    #                         hiddens=128,
    #                         use_best_strings=True,
    #                         rtr = False
    #                         )
    for i in range(0,1):
        corruption_level = 0.1
        name = "sphere_{0}".format(corruption_level)
        l = AEContinuousSolver()
        z=l.k_means(name,
                            pop_size=1000,
                            genome_length=50,
                            lim_percentage=20,
                            lim=20,
                            trials=1,
                            corruption_level=corruption_level,
                            num_epochs=25,
                            lr = 0.01,
                            online_training=True,
                            pickle_data=False,
                            save_data=False,
                            max_evaluations=500000,
                            cross_rate=1.0,
                            unique_training=True,
                            sample_rate=1,
                            hiddens=[100],
                            use_best_strings=False,
                            w=50,
                            sample_sd=0.01
                            )


    # args = sys.argv
    # pop_size=int(args[1])
    # print pop_size
    # genome_length=50
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
    # unique_training=int(args[7])
    # if unique_training == 0:
    #     unique_training = False
    # else:
    #     unique_training = True
    # pickle_data=False
    # sample_rate=int(args[8])
    # hiddens=int(args[9])
    # corruption_level=float(args[10])
    # use_best_strings=int(args[11])
    # if use_best_strings == 0:
    #     use_best_strings = False
    # else:
    #     use_best_strings = True
    # w=int(args[12])
    # sample_sd=float(args[13])
    # trial = int(args[14])
    # name = "ae_sphere_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}".format(pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,hiddens,corruption_level,use_best_strings,w,sample_sd,trial)
    # l = AEContinuousSolver()
    # z=l.not_probabilistic_ia(name,
    #                     pop_size=pop_size,
    #                     genome_length=genome_length,
    #                     lim_percentage=lim_percentage,
    #                     lim=20,
    #                     trials=1,
    #                     corruption_level=corruption_level,
    #                     num_epochs=num_epochs,
    #                     lr = lr,
    #                     online_training=online_training,
    #                     pickle_data=False,
    #                     save_data=False,
    #                     max_evaluations=100000,
    #                     cross_rate=1.0,
    #                     unique_training=unique_training,
    #                     sample_rate=1,
    #                     hiddens=[hiddens],
    #                     use_best_strings=use_best_strings,
    #                     w=w,
    #                     sample_sd=sample_sd
    #                     )

