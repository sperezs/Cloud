# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:13:02 2019

@author: sebas
"""

### This module contains functions for simulation
### Includes projection
### INcludes generate_random_loads


import numpy as np

## projection alg on Delta_eps
# projects y on Delta_eps

def projection(eps,N,y):
    
    sigma = sorted(range(len(y)), key=lambda k: y[k])
    
    y_temp = [ y[k] for k in sigma ]
    x_temp = []
    
    j = 0
    
    feasible = 0 
    
    while feasible < N:
        
        feasible = 0
        
        S = list(range(j))
        
        C = (1 - (eps/N)*j)/( sum( [y_temp[k] for k in range(j,N)] ) )
        
        x_temp = [eps/N for k in S] + [ C*y_temp[k] for k in range(j,N) ]
        
        
        for k in range(N):
            if x_temp[k]>= eps/N:
                feasible = feasible + 1
     
        j = j+1
        
    sigma_prime = sorted(range(len(sigma)), key=lambda k: sigma[k])
    x = [ x_temp[k] for k in sigma_prime]
    
    return x

## end projection
    

## Make Gamma loads

def make_load_gamma(N,beta,active,shape,bern,err):
    Load = {}
    for i in range(N):
        if i in active: 
            if np.random.rand() < bern:
                
                Load[i] = np.random.gamma(shape,beta[i]/(sum([beta[j] for j in active])*bern*shape))*(1-err)
            else:
                Load[i] = 0
        else:
            Load[i] = 0
    return Load
 
## end Gamma loads
    

### Simulator
    
# Input:(Times,Periods,Active_Periods,N,eps,eta,beta,P,S,lamb)
# Times = list of times we want to sample, by default is step , 2step, 3 step, ...
# Periods = partition of [0,Tmax] in periods
# Active_Periods = tells which users are active in each period
# N = Number of users
# eps = error parameter
# eta = eta parameter, usually 1/3
# beta = SLAs
# P = priorities, not used in this code
# S = initial distribution, every coordinate should be at least eps/N
# lamb = lamb parameter
# alpha = precision parameter
    

# Output
# work_online = overall work online done by proportional online mult weight alg indexed by Times (see above)
# Alg_work_out = individiual work done by online algorithm at any time, indexed by [t,i], t time, i user
# Opt_work_out_restricted = overall work done by optimal offline solution indexed by Times
# Opt_work_out = overall work done by (1-eps) optimal offline solution indexed by Times
# work_greedy_opt = individual work done by Proportional Greedy (see article)
# Queue_online = queues of online algorithm indexed by Times and users
# Queue_static = queues of static solution indexed by Times and users
# Queue_greedy_opt = queues of Proportional Greedy indexed by Times and users
# Queue_static_restricted = queues of constrained static solution indexed by Times and users



def mult_alg(Times,Periods,Active_Periods,N,eps,eta,beta,P,S,lamb,alpha):
    
    if len(Periods) != len(Active_Periods):
        print('not right amount of periods or active users periods')
        return
    
    T = max(Times)
    period = 0 ## current period
    
    
    H = {}
    
    
    # Initial distribution
    
    for j in range(N):
        H[0,j] = S[j]

    ### Output variables
    
    Alg_work_out = {}
    Opt_work_out = {}
    Opt_work_out_restricted = {}
    
    Queue_online = {}
    Queue_static = {}
    Queue_static_restricted = {}
    Queue_greedy_opt = {}
    Queue_greedy_opt_rest = {}
    
    ### variables inside algorithm
    
    work_online = {}
    work_static = {}
    work_static_restricted = {}
    work_greedy = {}
    work_greedy_opt = {}
    work_greedy_opt_rest = {}
    
    queue = {}
    queue_static = {}
    queue_static_restricted = {}
    queue_greedy = {}
    queue_greedy_opt = {}
    queue_greedy_opt_rest = {}

    for j in range(N):
        queue[j] = 0
        queue_static[j] = 0
        queue_static_restricted[j] = 0
        queue_greedy[j] = 0
        queue_greedy_opt[j] = 0
        queue_greedy_opt_rest[j] = 0

    work_alg = 0
    work_gre = 0
    work_opt = 0
    
    
    ### restriction of load and static allocations
    
    delta_1 = eps/N
    delta_2 =alpha*eps/N
    
    
    #######################
    # Multiplicative rule #
    #######################
    
    for t in range(T+1):
        
        Load = {}
        
        
        Load = make_load_gamma(N,beta,Active_Periods[period],2000,1,delta_1)
        
            
        
        A = []
        A_1 = []
        A_2 = []
        
        L = [0 for j in range(N)]
        
        remainder = 1-eps  ### used by greedy eps  (fair)
        
        for j in range(N):
            
            ### work algorithm
            work_online[t,j] = min( Load[j] + queue[j] , H[t,j]  )
            queue[j] = Load[j] + queue[j] - work_online[t,j]
            work_alg = work_alg + work_online[t,j]
            
            ### work greedy eps 
        
            work_greedy[j] = 0
            if remainder > 0:
                work_greedy[j] = min(Load[j] + queue_greedy[j],remainder)
                remainder = remainder - work_greedy[j]
            work_gre = work_gre + work_greedy[j]
            queue_greedy[j] = Load[j] + queue_greedy[j] - work_greedy[j]
            
            ### work static 
            
            ## set allocation for static
            
            h_static = 0
            if j in Active_Periods[period]:
                h_static = beta[j]/(sum(beta[k] for k in Active_Periods[period]))
                
                work_static[j] = min( Load[j] + queue_static[j] , h_static)
                queue_static[j] = Load[j] + queue_static[j] - work_static[j]
            
            work_static_restricted[j] = min( Load[j] + queue_static_restricted[j] , h_static*(1-delta_2)) 
            queue_static_restricted[j] = Load[j] + queue_static_restricted[j] - work_static_restricted[j]
            
            
            # Set active users
            if queue[j] > 0:
                A = A + [j]
        
        # update period
        if t in Periods:
            period = period + 1  ### move to next period
        
            
        ### work greedy fair 
        
        remainder_opt = 1  ### used by greedy opt  (fair)
        Remainder = {}
        
        ### Apply Proportional Greedy
        
        for i in range(N):
            Remainder[i] = queue_greedy_opt[i] + Load[i]
            work_greedy_opt[t,i] = 0
            
        while True:
            
            Active = []
            
            for i in range(N):
                
                if Remainder[i] > 0 and work_greedy_opt[t,i]<1:
                    Active = Active + [i]
        
            if len(Active) == 0 or remainder_opt <= 0:
                break

            Proportional_opt = sum([beta[i] for i in Active])
            
            i_min = min(Active)
            
            ### find minimizer
            for i in Active:
                if Remainder[i] < Remainder[i_min]:
                    i_min = i
                    
            if Remainder[i_min] < beta[i_min]*remainder_opt/Proportional_opt:
                work_greedy_opt[t,i_min] = work_greedy_opt[t,i_min] + Remainder[i_min]
                remainder_opt = remainder_opt - Remainder[i_min]
                Remainder[i_min] = 0
                
            else:
                for k in Active:
                    work_greedy_opt[t,k] = work_greedy_opt[t,k] + beta[k]*remainder_opt/Proportional_opt
                    Remainder[k] = Remainder[k] - beta[k]*remainder_opt/Proportional_opt
                remainder_opt = 0
            
        ## Update queues and work_greedy_opt
        for i in range(N):
            queue_greedy_opt[i] = Remainder[i]
            work_opt = work_opt + work_greedy_opt[t,i]
        
        
        ### work greedy fair (restricted) 
        
        remainder_opt_rest = 1 - delta_2  ### used by greedy opt  (fair)
        Remainder_rest = {}
        
        ### Apply Proportional Greedy
        
        for i in range(N):
            Remainder_rest[i] = queue_greedy_opt_rest[i] + Load[i]
            work_greedy_opt_rest[t,i] = 0
            
        while True:
            
            Active = []
            
            for i in range(N):
                
                if Remainder_rest[i] > 0 and work_greedy_opt_rest[t,i]<1:
                    Active = Active + [i]
        
            if len(Active) == 0 or remainder_opt_rest <= 0:
                break

            Proportional_opt = sum([beta[i] for i in Active])
            
            i_min = min(Active)
            
            ### find minimizer
            for i in Active:
                if Remainder_rest[i] < Remainder_rest[i_min]:
                    i_min = i
                    
            if Remainder_rest[i_min] < beta[i_min]*remainder_opt_rest/Proportional_opt:
                work_greedy_opt_rest[t,i_min] = work_greedy_opt_rest[t,i_min] + Remainder_rest[i_min]
                remainder_opt_rest = remainder_opt_rest - Remainder_rest[i_min]
                Remainder_rest[i_min] = 0
                
            else:
                for k in Active:
                    work_greedy_opt_rest[t,k] = work_greedy_opt_rest[t,k] + beta[k]*remainder_opt_rest/Proportional_opt
                    Remainder_rest[k] = Remainder_rest[k] - beta[k]*remainder_opt_rest/Proportional_opt
                remainder_opt_rest = 0
            
            
            
        ## Update queues and work_greedy_opt_rest
        for i in range(N):
            queue_greedy_opt_rest[i] = Remainder_rest[i]
        
        
        
        ### Update output in times given by Times
            
        if t in Times:
            print(t)
            Alg_work_out[t] = work_alg
                
            Opt_work_out[t] = work_opt
            Opt_work_out_restricted[t] = work_gre
    
            for i in range(N):
                Queue_online[t,i] = queue[i]
                Queue_static[t,i] = queue_static[i]
                Queue_static_restricted[t,i] = queue_static_restricted[i]
                Queue_greedy_opt[t,i] = Remainder[i]
                Queue_greedy_opt_rest[t,i] = Remainder_rest[i]
        
        # Update ### set loss function
        
        h_hat = []
    
        pmax = max([P[i] for i in A]+[0])
        
    
    ######################################### MODIFIED
    
        for i in A:
            if H[t,i] < (1-eps)*beta[i]/sum([beta[i] for i in A]):
                A_1 = A_1 + [i]
                L[i] = pmax + lamb
            else:
                A_2 = A_2 + [i]
                L[i] = P[i]
        
                
        # RULE 1 [Efficiency]
        
        
        for j in range(N):
            h_hat = h_hat + [H[t,j]*np.exp(eta*L[j])]
    
        # KL-projection
        h_hat = projection(eps,N,h_hat)
    
        for j in range(N):
            H[t+1,j] = h_hat[j]
    
    return H, work_online, Alg_work_out, Opt_work_out_restricted, Opt_work_out, work_greedy_opt, Queue_online, Queue_static, Queue_greedy_opt, Queue_static_restricted, Queue_greedy_opt_rest


### end algorithm
    
