from gurobipy import *
import random
import numpy
import csv
import argparse
import os

script_path = os.path.realpath(__file__)
default_folder_path = os.path.dirname(os.path.dirname(script_path))
default_gurobi_license_path = f"{os.path.expanduser('~')}/Downloads/gurobi.lic"

parser=argparse.ArgumentParser(
    description='''Trabalho computacional - otimização em redes. ''',
    epilog="""Alunos: Ana Júlia de Lima Martins e Samuel Souza da Silva.""")

parser.add_argument('--license', type=str, default=default_gurobi_license_path, help=f'Caminho para a licença gurobi (Default: {default_gurobi_license_path})')
parser.add_argument('--file', type=str, default=f"{default_folder_path}/data/sch100k1.csv", help=f'Caminho do arquivo csv (Default: {default_folder_path}/data/sch100k1.csv)')
parser.add_argument('--output_path', type=str, default='.', help=f'Caminho do arquivo de saída (Default: diretório corrente)')
parser.add_argument('--due_date', type=int, default=454, help=f'Tempo de entrega comum das tarefas (Default: 454)')
parser.add_argument('--verbose', type=bool, default=True, help=f'Habilitar modo verboso (Default: True)')
args=parser.parse_args()

# Access the converted values
license_path = args.license
input_filename = args.file
due_date = args.due_date
verbose = args.verbose
output_file_path = args.output_path

# Set the environment variable to point to your license file
os.environ["GRB_LICENSE_FILE"] = license_path

class CommonDueDateSchedulingModel:
    def __init__(self, due_date):
        self.D_d = due_date
        self.model = None

    def _initialize_model(self):
        if self.model is None:
            self.model = Model(name = 'Common Due Date Scheduling Problem')

            if verbose == False:
                self.model.setParam('OutputFlag', 0)

            self.model.setParam('TimeLimit', 10)
            

    def _add_input_variables(self, num_tasks):
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
    
    def get_model(self, num_tasks, p, alpha, beta, M):
        self._initialize_model()
        e, t, d, tau, J = self._add_input_variables(num_tasks)
        self._add_constraints(J, tau, d, M,  e, t, p, num_tasks)
        self._define_objective_function(alpha, e, beta, t, num_tasks)

        return self.model, e, t, d, tau, J

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

    def _order_task_by_delay_penalty(self, beta):
        # Create a list of tuples (index, penalty_for_delay)
        indexed_penalties = [(index+1, beta[index+1]) for index, task in enumerate(beta)]

        # Sort the list of tuples based on penalty_for_delay in descending order
        sorted_indexed_penalties = sorted(indexed_penalties, key=lambda x: x[1], reverse=True)

        # Extract the indexes from the sorted list of tuples
        sorted_tasks = [index for index, penalty in sorted_indexed_penalties]
        
        return sorted_tasks
    
    def _define_initial_condition(self, beta, alpha, J):
        beta_over_alpha = {key: beta[key] / alpha[key] for key in alpha}
        taks_sorted_by_delay_penalty = self._order_task_by_delay_penalty(beta_over_alpha)

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

    def _found_a_valid_solution(self):
        if self.model.SolCount > 0:
            if self.verbose:
                print("Best solution found:")
                for v in self.model.getVars():
                    print(f"{v.VarName} {v.X:g}")
            return True
        else:
            print("Error: it was not able to find a valid solution")
            exit(1)

    def _relax_and_fix(self, J, num_tasks):
        if self.verbose:
            print ("First iteration of relax and fix...")
        for i in range(0, num_tasks):
            task_id = i + 1
            for j in range(0, num_tasks):
                order = j +1
                if i < round(0.3 * num_tasks):
                    self._unfix_variable(J[task_id][order])
                else:
                    self._relax_variable(J[task_id][order])
        
        self.model.optimize()
        self._found_a_valid_solution()

        if self.verbose:
            print ("Second iteration of relax and fix...")
        for i in range(0, num_tasks):
            task_id = i + 1
            for j in range(0, num_tasks):
                order = j +1
                if i < round(0.3 * num_tasks):
                    self._fix_variable(J[task_id][order], J[task_id][order].X)
                elif i < round(0.6 * num_tasks) :
                    self._unrelax_variable(J[task_id][order], GRB.INTEGER)
                else:
                    self._relax_variable(J[task_id][order])
        
        self.model.optimize()
        self._found_a_valid_solution()

        if self.verbose:
            print ("Third iteration of relax and fix...")
        for i in range(0, num_tasks):
            task_id = i + 1
            for j in range(0, num_tasks):
                order = j +1
                if i < round(0.6 * num_tasks) :
                    self._fix_variable(J[task_id][order], J[task_id][order].X)
                else:
                    self._unrelax_variable(J[task_id][order], GRB.INTEGER)
        
        self.model.optimize()
        self._found_a_valid_solution()

    def _fix_and_optimize(self, J, num_tasks):
        if self.verbose:
            print ("First iteration of fix and optmize...")
        for i in range(0, num_tasks):
            task_id = i + 1
            for j in range(0, num_tasks):
                order = j +1
                if i < round(0.3 * num_tasks):
                    self._unfix_variable(J[task_id][order])
                else:
                    self._fix_variable(J[task_id][order], J[task_id][order].X)
        
        self.model.optimize()
        self._found_a_valid_solution()

        if self.verbose:
            print ("Second iteration of fix and optmize...")
        for i in range(0, num_tasks):
            task_id = i + 1
            for j in range(0, num_tasks):
                order = j +1
                if i < round(0.25 * num_tasks):
                    self._fix_variable(J[task_id][order], J[task_id][order].X)
                elif i < round(0.6 * num_tasks) :
                    self._unfix_variable(J[task_id][order])
                else:
                    self._fix_variable(J[task_id][order], J[task_id][order].X)
        
        self.model.optimize()
        self._found_a_valid_solution()

        if self.verbose:
            print ("Third iteration of fix and optmize...")
        for i in range(0, num_tasks):
            task_id = i + 1
            for j in range(0, num_tasks):
                order = j +1
                if i < round(0.55 * num_tasks) :
                    self._fix_variable(J[task_id][order], J[task_id][order].X)
                else:
                    self._unfix_variable(J[task_id][order])
        
        self.model.optimize()
        self._found_a_valid_solution()

    def _fix_variable(self, var, value):
        var.lb = value
        var.ub = value
        return var
    
    def _unfix_variable(self, var, lb=0):
        var.lb = lb
        var.ub = float('inf')
        return var
    
    def _unrelax_variable(self, var, vtype, lower_default=0):
        var.vtype = vtype
        var.lb = lower_default  # Reset lower bound to default (0)
        var.ub = float('inf')  # Reset upper bound to default (unconstrained)
        return var
    
    def _relax_variable(self, var):
        var.vtype = GRB.CONTINUOUS
        var.ub = float('inf')
        var.lb = 0
        return var

    def _get_delayed_and_early_tasks(self, t, num_tasks):
        delayed_tasks = []
        early_tasks = []
        for task_id in range(1, num_tasks+1):
            if t[task_id].X > 0:
                delayed_tasks.append(task_id)
            else:
                early_tasks.append(task_id)
        return delayed_tasks, early_tasks

    def _neighborhood_change(self, current_best_solution, candidate, k):
        current_best_solution_J, current_best_solution_e,current_best_solution_t,current_best_solution_model = current_best_solution
        candidate_J, candidate_e,candidate_t, candidate_model = candidate
        current_best_value = current_best_solution_model.ObjVal
        candidate_value = candidate_model.ObjVal

        if candidate_value < current_best_value:
            best_solution_model = candidate_model
            best_solution_J = candidate_J
            best_solution_e = candidate_e
            best_solution_t = candidate_t
            k = 1
        else:
            best_solution_model = current_best_solution_model
            best_solution_J = current_best_solution_J
            best_solution_e = current_best_solution_e
            best_solution_t = current_best_solution_t
            k = k + 1

        return best_solution_J, best_solution_e, best_solution_t, best_solution_model, k

    def _shake(self, x, k):
        if k == 1:
            return self._shake_on_neighborhood_1(x)
        elif k == 2:
            return self._shake_on_neighborhood_2(x)
        return self._shake_on_neighborhood_3(x)

    def _fix_J_Matrix_from_J_vector(self, J_vector, J_matrix, num_tasks):
        for task_id in range(1, num_tasks+1):
            for order in range(1, num_tasks+1):
                if task_id == J_vector[order-1]:
                    self._fix_variable(J_matrix[task_id][order], 1)
                else:
                    self._fix_variable(J_matrix[task_id][order], 0)

    def _get_array_from_J_matrix(self, J, num_tasks):
        converted_array = []

        for i in range(1, num_tasks+1):
            for j in range(1, num_tasks+1):
                if J[j][i].X == 1:
                    converted_array.append(j)
                    break
        
        return converted_array

    def _fix_current_J_matrix(self, J, num_tasks):
        for i in range(1, num_tasks+1):
            for j in range(1, num_tasks+1):
                self._fix_variable(J[i][j], J[i][j].X)

    def _fix_J_matrix(self, J_to_fix, J_to_copy, num_tasks):
        for i in range(1, num_tasks+1):
            for j in range(1, num_tasks+1):
                self._fix_variable(J_to_fix[i][j], J_to_copy[i][j].X)

    # Randomly select a delayed task and put in the first order
    def _shake_on_neighborhood_1(self, x):
        e, t, J = x
        num_tasks = len(e)

        delayed_tasks, _ = self._get_delayed_and_early_tasks(t, num_tasks)
        num_delayed_tasks = len(delayed_tasks)
        random_selected_delay_task_id = delayed_tasks[random.randint(0, num_delayed_tasks-1)]
        J_vector = self._get_array_from_J_matrix(J, num_tasks)

        # Put random selected delay task as first task to be complited
        J_vector.remove(random_selected_delay_task_id)
        J_vector = [random_selected_delay_task_id] + J_vector
        self._fix_J_Matrix_from_J_vector(J_vector, J, num_tasks)
        # Here we call model optimize only to compute objective function from new shaked solution
        self.model.optimize()

    # Randomly select a delayed task and randomly select a early task and switch them
    def _shake_on_neighborhood_2(self, x):
        e, t, J = x
        num_tasks = len(e)

        delayed_tasks, early_tasks = self._get_delayed_and_early_tasks(t, num_tasks)
        num_delayed_tasks = len(delayed_tasks)
        num_early_tasks = len(early_tasks)
        random_selected_delay_task_index = random.randint(0, num_delayed_tasks-1)
        random_selected_early_task_index = random.randint(0, num_early_tasks-1)

        J_vector = self._get_array_from_J_matrix(J, num_tasks)

        # Switch elements
        aux = J_vector[random_selected_delay_task_index]
        J_vector[random_selected_delay_task_index] = J_vector[random_selected_early_task_index]
        J_vector[random_selected_early_task_index] = aux

        self._fix_J_Matrix_from_J_vector(J_vector, J, num_tasks)
        # Here we call model optimize only to compute objective function from new shaked solution
        self.model.optimize()

    # Randomly select a early task and put in the last order
    def _shake_on_neighborhood_3(self, x):
        e, t, J = x
        num_tasks = len(e)

        _, early_tasks = self._get_delayed_and_early_tasks(t, num_tasks)
        num_early_tasks = len(early_tasks)
        random_selected_early_task_id = early_tasks[random.randint(0, num_early_tasks-1)]
        J_vector = self._get_array_from_J_matrix(J, num_tasks)

        # Put random selected early task in the last task to be complited
        J_vector.remove(random_selected_early_task_id)
        J_vector = J_vector + [random_selected_early_task_id]
        self._fix_J_Matrix_from_J_vector(J_vector, J, num_tasks)
        # Here we call model optimize only to compute objective function from new shaked solution
        self.model.optimize()

    def _BVNS(self, input, k_max, max_iterations):
        num_it = 0
        e, t, J, num_tasks, p, alpha, beta, M = input
        x = (e, t, J)
        
        best_solution_model, best_solution_e,best_solution_t,_,_, best_solution_J = CommonDueDateSchedulingModel(self.D_d).get_model(num_tasks, p, alpha, beta, M)
        self._fix_J_matrix(best_solution_J, J, num_tasks)
        # Just call optimize here to compute value of objective function
        best_solution_model.optimize()

        while num_it < max_iterations:
            k = 1
            while k <= k_max:
                self._shake(x, k)
                self._fix_and_optimize(J, num_tasks)  
                J, e, t, self.model, k = self._neighborhood_change((best_solution_J, best_solution_e,best_solution_t, best_solution_model), (J, e, t, self.model), k)
                self._fix_current_J_matrix(J, num_tasks)
                self.model.optimize()
                x = (e, t, J)
                best_solution_model, best_solution_e,best_solution_t,best_solution_d,best_solution_tau, best_solution_J = CommonDueDateSchedulingModel(self.D_d).get_model(num_tasks, p, alpha, beta, M)
                self._fix_J_matrix(best_solution_J, J, num_tasks)
                best_solution_model.optimize() # Just call optimize here to compute value of objective function with fixed J matrix
                num_it = num_it + 1
                if (num_it >= max_iterations):
                    break

        return best_solution_model, best_solution_e, best_solution_t, best_solution_d, best_solution_tau, best_solution_J

    def write_best_solution_found(self, J_matrix, num_tasks):
        with open(f"{output_file_path}/AnaSamuel.csv", "w") as file:
            J_vector = self._get_array_from_J_matrix(J_matrix, num_tasks)
            for task in J_vector:
                file.write(f"{task}")
                if task != J_vector[-1]:
                    file.write(f", ")

    def compute_solution(self):
        p, alpha, beta, M = self._read_tasks_from_csv(self.csv_filename)
        num_tasks = len(p)
        self.model, e, t, d, tau, J = CommonDueDateSchedulingModel(self.D_d).get_model(num_tasks, p, alpha, beta, M)
        
        
        if not self.verbose:
            print("\nVerbose mode disabled. This operation may take a while. Please, wait...")
        
        self._define_initial_condition(beta, alpha, J)
        self.model.optimize()
        _, _, _, _, _, best_solution_J = self._BVNS((e, t, J, num_tasks, p, alpha, beta, M), 3, 1)

        if self.verbose:
            print(f"Obj: {self.model.ObjVal:g}")

            for v in self.model.getVars():
                print(f"{v.VarName} {v.X:g}")

        self.write_best_solution_found(best_solution_J, num_tasks)

        self.model.dispose()

problem = CommonDueDateSchedulingProblem(input_filename, due_date, verbose=verbose)
problem.compute_solution()