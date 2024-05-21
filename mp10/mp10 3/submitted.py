'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np
import math

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    P = np.zeros((model.M, model.N, 4, model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            if model.R[r, c] == 1 or model.R[r, c] == -1:
                continue
            for a in range(4):
                if a == 0:
                    if c - 1 < 0 or model.W[r, c - 1] == True:
                        P[r, c, a, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, a, r, c - 1] += model.D[r, c, 0]
                    if r + 1 >= model.M or model.W[r + 1, c] == True:
                        P[r, c, a, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, a, r + 1, c] += model.D[r, c, 1]
                    if r - 1 < 0 or model.W[r - 1, c] == True:
                        P[r, c, a, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, a, r - 1, c] += model.D[r, c, 2]
                elif a == 2:
                    if c + 1 >= model.N or model.W[r, c + 1] == True:
                        P[r, c, a, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, a, r, c + 1] += model.D[r, c, 0]
                    if r + 1 >= model.M or model.W[r + 1, c] == True:
                        P[r, c, a, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, a, r + 1, c] += model.D[r, c, 2]
                    if r - 1 < 0 or model.W[r - 1, c] == True:
                        P[r, c, a, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, a, r - 1, c] += model.D[r, c, 1]
                elif a == 1:
                    if r - 1 < 0 or model.W[r - 1, c] == True:
                        P[r, c, a, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, a, r - 1, c] += model.D[r, c, 0]
                    if c + 1 >= model.N or model.W[r, c + 1] == True:
                        P[r, c, a, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, a, r, c + 1] += model.D[r, c, 2]
                    if c - 1 < 0 or model.W[r, c - 1] == True:
                        P[r, c, a, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, a, r, c - 1] += model.D[r, c, 1]
                elif a == 3:
                    if r + 1 >= model.M or model.W[r + 1, c] == True:
                        P[r, c, a, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, a, r + 1, c] += model.D[r, c, 0]
                    if c + 1 >= model.N or model.W[r, c + 1] == True:
                        P[r, c, a, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, a, r, c + 1] += model.D[r, c, 1]
                    if c - 1 < 0 or model.W[r, c - 1] == True:
                        P[r, c, a, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, a, r, c - 1] += model.D[r, c, 2]
    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros((model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            max_a = -math.inf
            for a in range(4):
                max_a = max(max_a, np.dot(P[r, c, a, :, :].reshape((1, model.M * model.N)), U_current.reshape((1, model.M * model.N)).T))
            U_next[r, c] = model.R[r, c] + model.gamma * max_a
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition(model)
    U = np.zeros((model.M, model.N))
    
    for i in range(100):
        U_next = compute_utility(model, U, P)
        converge = True
        for r in range(model.M):
            for c in range(model.N):
                if abs(U_next[r, c] - U[r, c]) >= epsilon:
                    converge = False
        
        U = U_next
        if converge == True:
            break
    return U

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    U_cur = np.zeros((model.M, model.N))
    for i in range(200):
        U = np.zeros((model.M, model.N))
        converge = True
        for r in range(model.M):
            for c in range(model.N):
                U[r, c] = model.R[r, c] + model.gamma * np.dot(model.FP[r, c, :, :].reshape((1, model.M * model.N)), U_cur.reshape((1, model.M * model.N)).T)
                if abs(U[r, c] - U_cur[r, c]) >= epsilon:
                    converge = False
        U_cur = U
        if converge:
            break
    return U