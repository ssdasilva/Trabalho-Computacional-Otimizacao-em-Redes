from gurobipy import *
import numpy as np
import csv

# Set the environment variable to point to your license file
os.environ["GRB_LICENSE_FILE"] = "/home/samuel/Downloads/gurobi.lic"

class Task:
    def __init__(self, id, processing_time, penalty_for_early_completion, penalty_for_delay):
        self.id = id
        self.processing_time = processing_time
        self.penalty_for_early_completion = penalty_for_early_completion
        self.penalty_for_delay = penalty_for_delay
    
    def __str__(self):
        return f'id: {self.id}; processing_time: {self.processing_time}; penalty_for_early_completion = {self.penalty_for_early_completion}; penalty_for_delay: {self.penalty_for_delay}'

        
class CommonDueDateSchedulingProblem:
    def __init__(self, csv_filename, due_date, verbose = False):
        self.csv_filename = csv_filename
        self.verbose = verbose
        self.D_d = due_date

    def _read_tasks_from_csv(self, filename):
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar=';')
            
            beta = {}
            alpha = {}
            p = {}

            task_id = 1
            M = 1
            for row in spamreader:
                p[task_id] = int(row[0])
                alpha[task_id] =  int(row[1])
                beta[task_id] = int(row[2].replace(';',''))
                M = M + p[task_id]
                
                if self.verbose:
                    print(f'i: {task_id}; p: {p[task_id]}; alpha = {alpha[task_id]}; beta: {beta[task_id]}')

                task_id = task_id + 1
        
        return p, alpha, beta, M
        
    def _add_input_variables(self, num_tasks):
        self.model = Model(name = 'Common Due Date Scheduling Problem')

        e = {}
        t = {}
        d = {}
        tau = {}
        J = {}

        for index in range(0, num_tasks):
            e[index+1] = self.model.addVar(vtype=GRB.INTEGER, name=f'e_{index+1}', lb = 0)
            t[index+1] = self.model.addVar(vtype=GRB.INTEGER, name=f't_{index+1}', lb = 0)
            d[index+1] = self.model.addVar(vtype=GRB.INTEGER, name=f'd_{index+1}', lb = 0)
            tau[index+1] = self.model.addVar(vtype=GRB.INTEGER, name=f'tau_{index+1}', lb = 0)

            J[index+1] = {}
            for k in range(0, num_tasks):
                J[index+1][k+1] = self.model.addVar(vtype=GRB.BINARY, name=f'J_{index+1}_{k+1}')
        
        return e, t, d, tau, J

    def _add_constraints(self, J, tau, d, M, e, t, p, num_tasks):
        for i in range(0, num_tasks):
            self.model.addConstr(quicksum(J[i+1][j+1] for j in range(0, num_tasks)) == 1)
            self.model.addConstr(quicksum(J[j+1][i+1] for j in range(0, num_tasks)) == 1)
            self.model.addConstr(t[i+1] >= d[i+1] - self.D_d)
            self.model.addConstr(e[i+1] >= self.D_d - d[i+1])

            for k in range(0, num_tasks):
                self.model.addConstr(d[i+1] >= tau[k+1] - (M*(1-J[i+1][k+1])))

        self.model.addConstr(tau[1] == quicksum(J[i+1][1]*p[i+1] for i in range(0, num_tasks)))

        for k in range(1, num_tasks):
            self.model.addConstr(tau[k+1] == tau[k] + quicksum(J[i+1][k+1]*p[i+1] for i in range(0, num_tasks)))

        self.model.addConstr(quicksum(tau[k+1] for k in range(0, num_tasks)) == quicksum(d[i+1] for i in range(0, num_tasks)))

    def _define_objective_function(self, alpha, e, beta, t, num_tasks):
        self.model.setObjective(
            quicksum(((alpha[i+1]*e[i+1]) + (beta[i+1]*t[i+1])) for i in range(0, num_tasks)),
            sense=GRB.MINIMIZE
        )

    def _solve_problem(self, obj_fn):
        self.model.set_objective('min', obj_fn)
        self.model.print_information()
        self.model.solve()
        self.model.print_solution()

    def _order_by_delay(self, tasks):
        # Create a list of tuples (index, penalty_for_delay)
        indexed_penalties = [(index, task.penalty_for_delay) for index, task in enumerate(tasks)]

        # Sort the list of tuples based on penalty_for_delay in descending order
        sorted_indexed_penalties = sorted(indexed_penalties, key=lambda x: x[1], reverse=True)

        # Extract the indexes from the sorted list of tuples
        sorted_indexes = [index for index, penalty in sorted_indexed_penalties]
        
        return sorted_indexes
    
    def _define_initial_condition(self, tasks):
        sorted_indexes = self._order_by_delay(tasks)
        initial_J_matrix = [[0 for _ in range(len(tasks))] for _ in range(len(tasks))]

        for i in range(0, len(tasks)):
            for k in range(0, len(tasks)):
                if sorted_indexes[k] == i:
                    initial_J_matrix[i][k] = 1
                    print(i,k)

        if self.verbose:
            print(sorted_indexes)

        return initial_J_matrix
    
    def _relax_and_fix(self, tasks, J_matrix, obj_fn):
        self.fixed_constraints = []
        initial_J_matrix = self._define_initial_condition(tasks)
        for i in range(0, len(tasks)):
            for k in range(0, len(tasks)):
                self.fixed_constraints.append(self.model.add_constraint(J_matrix[i][k]==initial_J_matrix[i][k]))
        self._solve_problem(obj_fn)


    def compute_best_solution(self):
        p, alpha, beta, M = self._read_tasks_from_csv(self.csv_filename)
        num_tasks = len(p)
        e, t, d, tau, J = self._add_input_variables(num_tasks)
        self._add_constraints(J, tau, d, M,  e, t, p, num_tasks)
        obj_fn = self._define_objective_function(alpha, e, beta, t, num_tasks)
        
        self.model.optimize()

        for v in self.model.getVars():
            print(f"{v.VarName} {round(v.X):g}")

        print(f"Obj: {self.model.ObjVal:g}")
        #self._define_initial_condition(tasks)
        #self._relax_and_fix( tasks, J_matrix, obj_fn)

due_date = 2
input_filename = '../data/sch100k1.csv'
problem = CommonDueDateSchedulingProblem(input_filename, due_date, verbose=True)
problem.compute_best_solution()