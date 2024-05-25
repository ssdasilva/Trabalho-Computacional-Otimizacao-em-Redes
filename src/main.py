from gurobipy import *
import numpy as np
import csv

# Set the environment variable to point to your license file
os.environ["GRB_LICENSE_FILE"] = "/home/samuel/Downloads/gurobi.lic"
       
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

    def _order_task_by_delay_penalty(self, beta):
        # Create a list of tuples (index, penalty_for_delay)
        indexed_penalties = [(index+1, beta[index+1]) for index, task in enumerate(beta)]

        # Sort the list of tuples based on penalty_for_delay in descending order
        sorted_indexed_penalties = sorted(indexed_penalties, key=lambda x: x[1], reverse=True)

        # Extract the indexes from the sorted list of tuples
        sorted_tasks = [index for index, penalty in sorted_indexed_penalties]
        
        return sorted_tasks
    
    def _define_initial_condition(self, beta, J):
        taks_sorted_by_delay_penalty = self._order_task_by_delay_penalty(beta)

        if self.verbose:
            print('Taks sorted by delay penalty:', taks_sorted_by_delay_penalty)

        for i in range(0, len(beta)):
            task_id = i + 1
            for j in range(0, len(beta)):
                order = j +1
                if task_id == taks_sorted_by_delay_penalty[j]:
                    self._fix_variable(J[task_id][order], 1)
                else:
                    self._fix_variable(J[task_id][order], 0)

    
    def _relax_and_fix(self, tasks, J_matrix, obj_fn):
        #TO DO
        return


    def _fix_variable(self, var, value):
        var.lb = value
        var.ub = value
        return var
    
    def _unfix_variable(self, var, lower_default=0):
        var.lb = lower_default  # Reset lower bound to default (0)
        var.ub = None  # Reset upper bound to default (unconstrained)
        return var
    
    def _relax_variable(self, var):
        var.vtype = GRB.CONTINUOUS
        var.ub = None
        var.lb = None
        return var

    def compute_best_solution(self):
        p, alpha, beta, M = self._read_tasks_from_csv(self.csv_filename)
        num_tasks = len(p)
        e, t, d, tau, J = self._add_input_variables(num_tasks)
        self._add_constraints(J, tau, d, M,  e, t, p, num_tasks)
        obj_fn = self._define_objective_function(alpha, e, beta, t, num_tasks)
        #self._define_initial_condition(beta, J)
        self.model.optimize()

        for v in self.model.getVars():
            print(f"{v.VarName} {round(v.X):g}")

        print(f"Obj: {self.model.ObjVal:g}")
        #self._relax_and_fix( tasks, J_matrix, obj_fn)
        self.model.dispose()

due_date = 2
#input_filename = '../data/sch100k1.csv'
input_filename = '../data/simple.csv'
problem = CommonDueDateSchedulingProblem(input_filename, due_date, verbose=True)
problem.compute_best_solution()