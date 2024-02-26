import time
from fuzzer.runtime_dictionary import RuntimeDictionary
from model.operation_dependency_graph import OperationDependencyGraph
import uuid
from ._gene_bank import GeneBank
from ._gene import Gene
from ._chromosome import Chromosome
from .Mutator._add_mutator import AddMutator
from .Mutator._delete_mutator import DeleteMutator
from .Mutator._change_mutator import ChangeMutator
import copy
import itertools
import numpy as np


class GAGenerator:
    def __init__(self, apis=[], specification={}, odg: OperationDependencyGraph = OperationDependencyGraph(),
                 host_address="", pre_defined_headers={},
                 time_budget=300):
        self.runtime_dict = RuntimeDictionary()
        self.apis = apis
        self.server_address = host_address
        self.protocol = "HTTPS"
        self.sequences = odg.generate_sequence(simple=True)
        self.specification = specification
        self.start_time = time.time()
        # for test case limit time
        self.time_limit = 2000
        self.end_time = self.start_time + time_budget
        self.seed_id = uuid.uuid4().__str__()
        self.sequence_id = uuid.uuid4().__str__()
        self.iteration_id = 0
        self.request_count = 0
        self.response_count = 0
        self.violations = set()
        self.error_sequence = []
        self.pending_add_sequence = set()
        self.pending_remove_sequence = set()
        self.pending_test_sequence_to_remove = set()
        self.status_code_status = {}
        self.time_budget = time_budget
        self.send_msg_count = 0
        self.receive_msg_count = 0
        self.success_sequence_count = 0
        self.success_apis = set()
        self.error_apis = set()
        self.total_apis = set()
        self.success_sequence = set()
        self.begin = time.time()
        self.success_sequence_output = []
        self.pre_defined_headers = pre_defined_headers
        self.gene_bank = self._build_gene_bank()
        self.api_curve = [{
            "time": 0,
            'count': 0
        }]

    def _build_gene_bank(self):
        """
        Build bank of gene
        Returns: genes
        """
        gene_bank = GeneBank()
        for node in self.sequences:
            gene_bank.genes.add(node)
        return gene_bank.genes

    def _init_chromosomes(self, max_chromosome_size, initial_gene_size):
        """
        Initial  chromosomes
        Args:
            max_chromosome_size: max size of genes can be loaded into chromosome
            initial_gene_size: size of chromosomes
        Returns: Generated chromosomes
        """
        chromosomes = []
        for i in range(max_chromosome_size):
            genes = list(np.random.choice(self._gene_bank, size=initial_gene_size, replace=False))
            chromosomes.append(Chromosome(genes))
        return chromosomes

    def _init_mutators(self):
        """
        Returns: mutator for evolution
        """
        mutators = list()
        mutators.append(AddMutator("Add mutator"))
        mutators.append(DeleteMutator("Delete mutator"))
        mutators.append(ChangeMutator("Change mutator"))
        return mutators

    def run(self):
        """
        Main loop for MostCorrelation
        Args:
            time_budget: time for computation
            iteration: max iteration time
        Returns: best result
        """
        counter = 0
        start_time = time.time()
        while start_time + self.time_budget > time.time():
            self._chromosomes = self._evolve(self._chromosomes)
            for chromosome in self._chromosomes:
                self.result.append(copy.deepcopy(chromosome))
            if self.verbose:
                print(
                    f'time: {time.time()}, round:{counter}, best result:{sorted(self._chromosomes, key=lambda x: x.fitness)[0].fitness}')
            counter += 1
        return self.result

    def _evolve(self, chromosomes=[]):
        """
        iteration evolving
        Args:
            chromosomes: population
        Returns: generated chromosomes
        """
        for chromosome in chromosomes:
            chromosome.fitness = self._fitness(chromosome.genes)
        chromosomes = self._selection(chromosomes)
        chromosomes = self._crossover(chromosomes)
        self._mutation(chromosomes)
        for chromosome in chromosomes:
            chromosome.fitness = self._fitness(chromosome.genes)
        return chromosomes

    def _fitness(self, genes=[]):
        """
        Fitness function for chromosomes
        Args:
            genes: current population
        Returns: calculated chromosomes
        """
        edges = itertools.combinations(genes, 2)
        node_num = len(genes)
        fitness = 0
        for item in edges:
            try:
                fitness += float(self._graph[item[0]][item[1]]["weight"])
            except:
                pass
        fitness /= (
                (node_num - 1) * node_num)
        return fitness

    def _mutation(self, chromosomes=[]):
        """
        Apply mutation for chromosomes
        Args:
            chromosomes: chromosomes in current evolution
        Returns: mutated chromosomes
        """
        mutation_prob = 0.2
        for chromosome in chromosomes:
            if np.random.random() < mutation_prob:
                applied_mutators = []
                for mutator in self._mutators:
                    if mutator.can_mutate(self, chromosome):
                        applied_mutators.append(mutator)
                if len(applied_mutators) == 0:
                    continue
                mutator = np.random.choice(applied_mutators)
                chromosome.genes = mutator.mutate(self, chromosome)

    def _selection(self, chromosomes=[]):
        """
        Since there is no strategies for selection
        Args:
            chromosomes: current chromosomes
        Returns: selection of chromosomes
        """
        return chromosomes

    def _crossover(self, chromosomes=[]):
        """
        Args:
            chromosomes: list of chromosomes
        Returns: chromosomes after crossover
        """
        father = chromosomes[:len(chromosomes) // 2]
        mother = chromosomes[len(chromosomes) // 2:]
        crossover_range = min(len(father), len(mother))
        crossover_prob = 0.8
        new_population = []
        for i in range(crossover_range):
            if np.random.rand() < crossover_prob:
                son = mother[i]
                daughter = father[i]
            else:
                daughter = mother[i]
                son = father[i]
            new_population.append(son)
            new_population.append(daughter)
        # check if not even
        if len(chromosomes) % 2 == 1:
            rest = chromosomes[crossover_range << 1:]
            if np.random.rand() < 0.5:
                new_population.extend(rest)
            else:
                rest.extend(new_population)
                new_population = rest
        return new_population

    def get_result(self):
        """
        Returns: rest of MostCorrelation
        """
        iter_set = set()
        res = []
        for chromosome in self.result:
            elem = tuple(sorted(chromosome))
            if iter_set.__contains__(elem):
                continue
            res.append(chromosome)
            iter_set.add(elem)
        res = sorted(res, key=lambda x: -x.fitness)
        return [[gene for gene in item.genes] for item in res]
