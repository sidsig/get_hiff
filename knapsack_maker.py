import random,re,pickle
total_items = 100 
weight_range = (10,101)
value_range = (5,151)

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


def write(total_items,weight_range,value_range):
    string = ""
    for i in range(0,total_items):
        weight = int(random.uniform(*weight_range))
        value = int(random.uniform(*value_range))
        string += "id {0}    weight {1}    value {2}".format(i,weight,value)
        string += "\n"
    f = open('knapsacks/knapsack_{0}.dat'.format(total_items),'w')
    f.write(string)
    f.close()


# re.match('^front_x = (\[.*\]);$',lines):
#                         front_x = eval(re.match('front_x = (\[.*\]);.*',lines).group(1))


def read_knap(number=100):
    knap = []
    with open('knapsacks/knapsack_'+str(number)+'.dat') as f:
        for lines in f.read().split('\n'):
            # print lines
            r = re.match('id ([0-9]+)    weight ([0-9]+)    value ([0-9]+).*',lines)
            if r:
                id = r.group(1)
                weight = int(r.group(2))
                value = int(r.group(3))
                knap.append([id,weight,value])
    return knap

def read_knap_for_class(number=100):
    knap = []
    values = []
    weights = []
    with open('knapsacks/knapsack_'+str(number)+'.dat') as f:
        for lines in f.read().split('\n'):
            # print lines
            r = re.match('id ([0-9]+)    weight ([0-9]+)    value ([0-9]+).*',lines)
            if r:
                id = r.group(1)
                weight = int(r.group(2))
                value = int(r.group(3))
                knap.append([id,weight,value])
                values.append(value)
                weights.append(weight)
    k=KnapsackData("knapsack_"+str(number), 1, number, values, [2000], [weights])
    pickle.dump(k,open("knapsack_"+str(number)+".pkl","wb"))
    return knap

#k = read_knap()
if __name__ == '__main__':
    # write(500,weight_range,value_range)
    read_knap_for_class(500)
