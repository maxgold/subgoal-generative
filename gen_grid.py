import numpy as np


rob_grid  = np.zeros([9,9])
goal_grid = np.zeros([9,9])
rob_pos   = np.array([1,1])
goal_pos  = np.array([4,4])
goal_x    = goal_pos[0]
goal_y    = goal_pos[1]

at_goal = False

while !at_goal:
	cur_x = rob_pos[0]
	cur_y = rob_pos[1]
	if cur_x < goal_x:
		cur_x += 1












