###############################
#
# Created by Patrik Valkovic
# 03.04.21
#
###############################
import torch as t

class SATProblem:
    def __init__(self, vars, clauses, device = None, dtype=t.long):
        self._nvars = vars
        self._nclauses = len(clauses)
        self._clausses = clauses
        self._max_vars_per_clause = max(list(map(len, clauses)))
        self._indices = t.zeros((self._nclauses,self._max_vars_per_clause), dtype=t.long)
        for ci, clause in enumerate(self._clausses):
            for i, v in enumerate(clause):
                self._indices[ci,i] = (v-1) if v > 0 else (abs(v)-1+self._nvars)
            for i in range(i+1, self._max_vars_per_clause):
                self._indices[ci,i] = self._indices[ci,0]
        self._indices = self._indices.to(dtype).to(device)
        
    @property
    def nvars(self):
        return self._nvars

    @staticmethod
    def from_cnf_file(filename, device = None, dtype=t.long):
        vars = -1
        nclauses = -1
        clauses = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('c') or line.startswith('%'):
                    continue
                if line.startswith('p'):
                    assert 'p cnf' in line
                    vars, nclauses = line.split('cnf')[1].strip().split()
                    vars, nclauses = int(vars), int(nclauses)
                    print(f"Problem with {vars} variables and {nclauses} clauses")
                else:
                    clause = list(map(int, line.split()))
                    clause = list(filter(lambda x: x != 0, clause))
                    if len(clause) == 0:
                        continue
                    clauses.append(clause)
        assert len(clauses) == nclauses
        return SATProblem(vars, clauses, device, dtype)

    def fitness_count_satisfied(self, population, *args, **kwargs):
        variables = t.cat([population, t.logical_not(population)], dim=1)
        satisfied = t.any(variables[:, self._indices], dim=2)
        num_satisfied = t.count_nonzero(satisfied, dim=1)
        return num_satisfied.to(t.float32)

    def fitness_count_unsatisfied(self, *args, **kwargs):
        return self._nclauses - self.fitness_count_satisfied(*args, **kwargs)
