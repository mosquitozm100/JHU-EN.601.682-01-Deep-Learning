import os
import math
import numpy as np

P = np.array([1,6,12,5,2,8,12,4])
Q = np.array([1,3,6,8,15,10,5,2])

def comp_distribution(P):
    #print(np.sum(P))
    #print(P.shape)
    p = P.astype(np.float64) / np.sum(P)
    #print(p)
    return p

def comp_KL(p, q):
    return  np.dot(p, ((np.log(p / q)).T))

def print_in_format(p):
    print('[', end = '')
    for i in p:
        print(i,end=',\  ')
    print("]")
    
#print("Good Morning!", end = '')
#print("What a wonderful day!") 

if __name__ == "__main__":
    #print('P:', P)
    p = comp_distribution(P)
    q = comp_distribution(Q)

    #print('p:', p)
    #print('q:', q)
    print_in_format(p)
    print_in_format(q)

    KL_p_q = comp_KL(p, q)
    KL_q_p = comp_KL(q, p)

    print('KL_p_q:', KL_p_q)
    print('KL_q_p:', KL_q_p)

