from docplex.mp.model import Model
import numpy as np
import csv

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
        self.due_date = due_date

    def _get_tasks(self, filename):
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar=';')
            
            tasks = []
            task_id = 1
            M = 1
            for row in spamreader:
                processing_time = int(row[0])
                penalty_for_early_completion =  int(row[1])
                penalty_for_delay = int(row[2].replace(';',''))
                task = Task(task_id, processing_time, penalty_for_early_completion, penalty_for_delay)
                tasks.append(task)
                task_id = task_id + 1
                M = M + processing_time

                if self.verbose:
                    print(task)
        
        return tasks, M
        
    def _add_input_variables_to_milp(self, tasks):
        self.milp_model = Model(name = 'MILP')
        early_completion_times = []
        delay_times = []
        dates_of_delivery = []
        processing_time_of_order_k = []

        num_tasks = len(tasks)
        J_matrix = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]
        index = 0

        for task in tasks:
            early_completion_times.append(self.milp_model.integer_var(name=f'e_{task.id}', lb = 0))
            delay_times.append(self.milp_model.integer_var(name=f't_{task.id}', lb = 0))
            dates_of_delivery.append(self.milp_model.integer_var(name=f'd_{task.id}', lb = 0))
            processing_time_of_order_k.append(self.milp_model.integer_var(name=f'tau_{index+1}', lb = 0))

            for k in range(0, num_tasks):
                J_matrix[index][k] = self.milp_model.binary_var(name=f'J_{index+1}_{k+1}')

            index = index + 1
        
        return early_completion_times, delay_times, dates_of_delivery, processing_time_of_order_k, J_matrix

    def _add_constraints_to_milp(self, tasks, J_matrix, processing_time_of_order_k, dates_of_delivery, M, early_completion_times, delay_times):
        tau_k_constraints = [0 for _ in range(len(tasks))]
        tau_sum = 0
        dates_of_delivery_sum = 0
        minimum_delay_of_task = [0 for _ in range(len(tasks))]
        minimum_early_completion_of_task = [0 for _ in range(len(tasks))]

        for task in tasks:
            task_id = task.id
            i = task_id - 1
            task_order = i
            J_sum_horizontal = 0
            J_sum_vertical = 0
            minimum_delivery_dates_of_task = [[0 for _ in range(len(tasks))] for _ in range(len(tasks))]

            for k in range(0, len(tasks)):
                J_sum_horizontal =  J_sum_horizontal + J_matrix[i][k] 
                J_sum_vertical = J_sum_vertical + J_matrix[k][i] 
                tau_k_constraints[task_order] = tau_k_constraints[task_order] + (J_matrix[k][task_order] * tasks[k].processing_time)

            if task_order != 0:
                tau_k_constraints[task_order] = tau_k_constraints[task_order] + tau_k_constraints[task_order - 1]

            tau_sum = tau_sum + tau_k_constraints[task_order]
            dates_of_delivery_sum = dates_of_delivery_sum + dates_of_delivery[i]
            minimum_delay_of_task[i] = dates_of_delivery[i] - self.due_date
            minimum_early_completion_of_task[i] = self.due_date - dates_of_delivery[i]

            self.milp_model.add_constraint(J_sum_horizontal==1)
            self.milp_model.add_constraint(J_sum_vertical==1)

            for k in range(0, len(tasks)):
                minimum_delivery_dates_of_task[i][k] = tau_k_constraints[k] - (M * (1 - J_matrix[i][k]))
                self.milp_model.add_constraint(dates_of_delivery[i] >= minimum_delivery_dates_of_task[i][k])


        for index, value in enumerate(tau_k_constraints):
            self.milp_model.add_constraint(tau_k_constraints[index]==processing_time_of_order_k[index])
            self.milp_model.add_constraint(early_completion_times[index] >= minimum_early_completion_of_task[index])
            self.milp_model.add_constraint(delay_times[index] >= minimum_delay_of_task[index])

        self.milp_model.add_constraint(tau_sum==dates_of_delivery_sum)


    def _define_objective_function(self, tasks, early_completion_times, delay_times):
        obj_fn = 0

        for task in tasks:
            i = task.id - 1
            obj_fn = obj_fn + (task.penalty_for_early_completion * early_completion_times[i]) + (task.penalty_for_delay * delay_times[i])

        return obj_fn

    def _solve_problem(self, obj_fn):
        self.milp_model.set_objective('min', obj_fn)
        self.milp_model.print_information()
        self.milp_model.solve()
        self.milp_model.print_solution()

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
                self.fixed_constraints.append(self.milp_model.add_constraint(J_matrix[i][k]==initial_J_matrix[i][k]))
        self._solve_problem(obj_fn)


    def compute_best_solution(self):
        tasks, M = self._get_tasks(self.csv_filename)
        early_completion_times, delay_times, dates_of_delivery, processing_time_of_order_k, J_matrix = self._add_input_variables_to_milp(tasks)
        self._add_constraints_to_milp(tasks, J_matrix, processing_time_of_order_k, dates_of_delivery, M,  early_completion_times, delay_times)
        obj_fn = self._define_objective_function(tasks, early_completion_times, delay_times)
        self._define_initial_condition(tasks)
        self._relax_and_fix( tasks, J_matrix, obj_fn)

due_date = 2
input_filename = 'sch100k1.csv'
#input_filename = 'test.csv'
problem = CommonDueDateSchedulingProblem(input_filename, due_date, verbose=True)
problem.compute_best_solution()