from docplex.mp.model import Model

milp_model = Model(name = 'MILP')

p_1 = 3
p_2 = 5
D_d = 2

e_1 = milp_model.integer_var(name='e_1', lb = 0)
t_1 = milp_model.integer_var(name='t_1', lb = 0)
e_2 = milp_model.integer_var(name='e_2', lb = 0)
t_2 = milp_model.integer_var(name='t_2', lb = 0)

d_1 = milp_model.integer_var(name='d_1', lb = 0)
d_2 = milp_model.integer_var(name='d_2', lb = 0)


J_11 = milp_model.binary_var(name='J_11')
J_12 = milp_model.binary_var(name='J_12')
J_21 = milp_model.binary_var(name='J_21')
J_22 = milp_model.binary_var(name='J_22')

tau_1 = milp_model.integer_var(name='tau_1', lb = 0)
tau_2 = milp_model.integer_var(name='tau_2', lb = 0)

c1 = milp_model.add_constraint(J_11 + J_21==1)
c2 = milp_model.add_constraint(J_12 + J_22==1)
c3 = milp_model.add_constraint(J_11 + J_12==1)
c4 = milp_model.add_constraint(J_21 + J_22==1)

c5 = milp_model.add_constraint(((J_11 * p_1) + (J_21 * p_2)) == tau_1)
c6 = milp_model.add_constraint((tau_1 + ((J_12 * p_1) + (J_22 * p_2))) == tau_2)

M=p_1+p_2+1
#c7 = milp_model.add_constraint(d_1 == (tau_1 * J_11) + (tau_2 * J_12) )
c7 = milp_model.add_constraint(d_1 >= tau_1 - (M * (1-J_11)) )
c8 = milp_model.add_constraint(d_1 >= tau_2 - (M * (1-J_12)) )
c9 = milp_model.add_constraint(d_2 >= tau_1 - (M * (1-J_21)) )
c10 = milp_model.add_constraint(d_2 >= tau_2 - (M * (1-J_22)) )
c11 = milp_model.add_constraint(tau_1 + tau_2 == d_1 + d_2 )
obj_fn = (20 * e_1) + (80 * t_1) + (30 * e_2) + (700 * t_2)


c12 = milp_model.add_constraint(t_1 >= d_1 - D_d)
c13 = milp_model.add_constraint(t_2 >= d_2 - D_d)
c14 = milp_model.add_constraint(e_1 >= D_d - d_1)
c15 = milp_model.add_constraint(e_2 >= D_d - d_2)
milp_model.set_objective('min', obj_fn)
milp_model.print_information()
milp_model.solve()
milp_model.print_solution()