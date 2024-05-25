from gurobipy import *

milp_model = Model(name = 'MILP')

p_1 = 3
p_2 = 5
D_d = 2

e_1 = milp_model.addVar(vtype=GRB.INTEGER, name='e_1', lb = 0)
t_1 = milp_model.addVar(vtype=GRB.INTEGER, name='t_1', lb = 0)
e_2 = milp_model.addVar(vtype=GRB.INTEGER, name='e_2', lb = 0)
t_2 = milp_model.addVar(vtype=GRB.INTEGER, name='t_2', lb = 0)

d_1 = milp_model.addVar(vtype=GRB.INTEGER, name='d_1', lb = 0)
d_2 = milp_model.addVar(vtype=GRB.INTEGER, name='d_2', lb = 0)


J_11 = milp_model.addVar(vtype=GRB.BINARY, name='J_11')
J_12 = milp_model.addVar(vtype=GRB.BINARY, name='J_12')
J_21 = milp_model.addVar(vtype=GRB.BINARY, name='J_21')
J_22 = milp_model.addVar(vtype=GRB.BINARY, name='J_22')

tau_1 = milp_model.addVar(vtype=GRB.INTEGER, name='tau_1', lb = 0)
tau_2 = milp_model.addVar(vtype=GRB.INTEGER, name='tau_2', lb = 0)

c1 = milp_model.addConstr(J_11 + J_21==1)
c2 = milp_model.addConstr(J_12 + J_22==1)
c3 = milp_model.addConstr(J_11 + J_12==1)
c4 = milp_model.addConstr(J_21 + J_22==1)

c5 = milp_model.addConstr(((J_11 * p_1) + (J_21 * p_2)) == tau_1)
c6 = milp_model.addConstr((tau_1 + ((J_12 * p_1) + (J_22 * p_2))) == tau_2)

M=p_1+p_2+1
#c7 = milp_model.addConstr(d_1 == (tau_1 * J_11) + (tau_2 * J_12) )
c7 = milp_model.addConstr(d_1 >= tau_1 - (M * (1-J_11)) )
c8 = milp_model.addConstr(d_1 >= tau_2 - (M * (1-J_12)) )
c9 = milp_model.addConstr(d_2 >= tau_1 - (M * (1-J_21)) )
c10 = milp_model.addConstr(d_2 >= tau_2 - (M * (1-J_22)) )
c11 = milp_model.addConstr(tau_1 + tau_2 == d_1 + d_2 )
obj_fn = (20 * e_1) + (80 * t_1) + (30 * e_2) + (700 * t_2)


c12 = milp_model.addConstr(t_1 >= d_1 - D_d)
c13 = milp_model.addConstr(t_2 >= d_2 - D_d)
c14 = milp_model.addConstr(e_1 >= D_d - d_1)
c15 = milp_model.addConstr(e_2 >= D_d - d_2)
milp_model.setObjective(obj_fn, sense=GRB.MINIMIZE)
milp_model.optimize()

for v in milp_model.getVars():
        print(f"{v.VarName} {v.X:g}")

print(f"Obj: {milp_model.ObjVal:g}")
print(type(J_11))