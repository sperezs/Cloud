# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:26:51 2019

@author: sebastian
"""

## imports

import matplotlib.pyplot as plt
import numpy as np
import time

#import Functions

from Functions import mult_alg


## end imports


#############
### Main ####
#############
    
## parameters


N = 3
Tmax = 3000000
step = 100000
eps = 0.01
eta = 0.05
init_dist = [1/N for i in range(N)]
beta = [0.2,0.3, 0.5]
Periods = [ int(Tmax/6) , int(2*Tmax/6),int(3*Tmax/6), int(4*Tmax/6), int(5*Tmax/6) , Tmax ]
Active_Period = [[1,2],[0,1],[0,2],[1,2],[0,1],[0,2]]


### do not modify the following parameters

lamb = (eps**2)/(8*N)
priorities = [1 for i in range(N)]
alpha = 1

## end parameters


Alg_work_out = {}
Queue_online = {}

Opt_work_out_restricted = {}
Opt_work_out = {}

Queue_static = {}


for T in range(0,Tmax+1,step):
    
    for i in range(N):
        Queue_online[T,i] = 0
        Queue_static[T,i] = 0
        
    Alg_work_out[T] = 0
    Opt_work_out_restricted[T] = 0
    Opt_work_out[T] = 0



## Run simulations
 
Times = range(0,Tmax+1,step)

start_time = time.clock()

allocation_online, work_online, Alg_work_out, Opt_work_out_restricted, Opt_work_out, work_greedy_opt, Queue_online, Queue_static, Queue_greedy_opt,Queue_static_restricted, Queue_greedy_opt_rest = mult_alg(Times,Periods,Active_Period,N,eps,eta,beta,priorities,init_dist,lamb,alpha)

print('running time', time.clock() - start_time)



#############

# Save Data #

#############


np.save('allocation_online.npy',allocation_online)
np.save('work_online.npy',work_online)
np.save('Alg_work_out.npy', Alg_work_out)
np.save('Opt_work_out_restricted.npy', Opt_work_out_restricted)
np.save('Opt_work_out.npy', Opt_work_out)
np.save('work_greedy_opt.npy', work_greedy_opt)
np.save('Queue_online.npy', Queue_online)
np.save('Queue_static.npy', Queue_static)
np.save('Queue_greedy_opt.npy', Queue_greedy_opt)
np.save('Queue_static_restricted.npy', Queue_static_restricted)
np.save('Queue_greedy_opt_rest.npy', Queue_greedy_opt_rest)


#############
### Plots ###
#############

scale = 10**(-6)

X = list(range(0,Tmax+1,step))
del X[0]


### Gap commparison
plt.figure(figsize=(10,5))
plt.plot([scale*T for T in X],[np.log10(Opt_work_out[T]-Alg_work_out[T]) for T in X],'b.--',linewidth=1,label='OPT - Alg')
plt.plot([scale*T for T in X],[np.log10(Alg_work_out[T]-Opt_work_out_restricted[T]) for T in X],'r.--',linewidth=1,label='Alg - (1-eps)-OPT')
plt.plot([scale*T for T in X],[np.log10(eps*T + 0) for T in X],'g.--',linewidth=1,label='log10(eps*T)')

#plt.plot([scale*T for T in range(0,Tmax+1,step)],[Opt_work_out[T]/Repetitions for T in range(0,Tmax+1,step)],'go--',linewidth=1,label='1-opt')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.xlabel("Time in millions")
plt.title('Log difference comparison')

plt.savefig('Char3users_log.pdf', bbox_inches='tight')

plt.show()


## end first plot N*np.log(N/eps)/(eta*eps**2)

plt.rcParams['agg.path.chunksize'] = 10000

#### Queue comparison versus static
for i in range(N):
    plt.figure(figsize=(10,5))
    plt.plot([scale*T for T in range(0,Tmax+1,step)],[Queue_online[T,i] for T in range(0,Tmax+1,step)],'b.--',linewidth=1.5,label='Algorithm')
    plt.plot([scale*T for T in range(0,Tmax+1,step)],[Queue_greedy_opt_rest[T,i] for T in range(0,Tmax+1,step)],'r.--',linewidth=1.5,label='Prop-Opt-rest')
    plt.plot([scale*T for T in range(0,Tmax+1,step)],[Queue_static_restricted[T,i] for T in range(0,Tmax+1,step)],'m.--',linewidth=1.5,label='(1-eps/3)-static')
    plt.plot([scale*T for T in range(0,Tmax+1,step)],[Queue_greedy_opt[T,i] for T in range(0,Tmax+1,step)],'g.--',linewidth=1.5,label='Prop-Opt')

    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    plt.xlabel("Time in millions")
    plt.title('queues user '+str(i+1))

    plt.savefig('Charqueues'+str(i)+'.pdf', bbox_inches='tight')

    plt.show()





### plot work

for i in range(N):
    
    plt.figure(figsize=(10,5))
    plt.plot([scale*T for T in range(Tmax+1)],[work_greedy_opt[T,i] for T in range(Tmax+1)],'g-',linewidth=1,label='Greedy prop')
    plt.plot([scale*T for T in range(Tmax+1)],[work_online[T,i] for T in range(Tmax+1)],'b-',linewidth=1.5,label='Algorithm')
    plt.plot([scale*T for T in range(0,Tmax+1,step)],[beta[i] for T in range(0,Tmax+1,step)],'r--',linewidth=1.5,label='SLA')
    
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    plt.ylim(-0.01, 1.04)
    plt.xlabel("Time in millions")
    plt.title('work user '+str(i+1))


    plt.savefig('work_user_'+str(i)+'.pdf', bbox_inches='tight')

    plt.show()

### plot allocation and work online


Y = range(1895,1905)
for i in range(N):
    
    plt.figure(figsize=(10,5))
    plt.plot([scale*T for T in Y],[allocation_online[T,i] for T in Y],'g-',linewidth=1,label='allocation')
    plt.plot([scale*T for T in Y],[work_online[T,i] for T in Y],'b-',linewidth=1.5,label='Algorithm')
#    plt.plot([scale*T for T in Y],[beta[i] for T in Y],'r--',linewidth=1.5,label='SLA')
    
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

 #   plt.ylim(-0.01, 1.04)
    plt.xlabel("Time in millions")
    plt.title('allocation and work online '+str(i+1))


    plt.savefig('allocation_work_user_'+str(i)+'.pdf', bbox_inches='tight')

    plt.show()


#### Compare cumulative works, online vs optimal greedy

Cum_work_online = {}
Cum_work_greedy_opt = {}

for i in range(N):
    Cum_temp_online = 0
    Cum_temp_offline = 0
    for t in range(Tmax+1):
        Cum_temp_online = Cum_temp_online + work_online[t,i]
        Cum_temp_offline = Cum_temp_offline + work_greedy_opt[t,i]
        
        if t in Times:
            Cum_work_online[t,i] = Cum_temp_online
            Cum_work_greedy_opt[t,i] = Cum_temp_offline



    
    
for i in range(N):
    plt.figure(figsize=(10,5))
    plt.plot([scale*T for T in range(0,Tmax+1,step)],[((Cum_work_greedy_opt[T,i]-Cum_work_online[T,i])) for T in range(0,Tmax+1,step)],'b.--',linewidth=1,label='Algorithm')

    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)


    plt.xlabel("Time in millions")
    plt.title('Cumulative work '+str(i+1))

    plt.savefig('Char_work_comparison'+str(i)+'.pdf', bbox_inches='tight')

    plt.show()


#### end work comparison



### plot square root of sum of squares of queues
    
    
plt.figure(figsize=(10,5))
plt.plot([scale*T for T in range(0,Tmax+1,step)],[np.sqrt(sum([Queue_online[T,i]**2 for i in range(N)])) for T in range(0,Tmax+1,step)],'b.--',linewidth=1,label='Algorithm')
plt.plot([scale*T for T in range(0,Tmax+1,step)],[np.sqrt(sum([Queue_greedy_opt[T,i]**2 for i in range(N) ])) for T in range(0,Tmax+1,step)],'g.--',linewidth=1,label='Prop-Opt')
plt.plot([scale*T for T in range(0,Tmax+1,step)],[np.sqrt(sum([Queue_greedy_opt_rest[T,i]**2 for i in range(N) ])) for T in range(0,Tmax+1,step)],'r.--',linewidth=1,label='Prop-Opt-rest')
plt.plot([scale*T for T in range(0,Tmax+1,step)],[np.sqrt(sum([Queue_static_restricted[T,i]**2 for i in range(N) ])) for T in range(0,Tmax+1,step)],'m.--',linewidth=1,label='(1-eps/3)-static')

plt.legend()

plt.xlabel("Time in millions")
plt.title('square queues')


plt.savefig('Queues_squared.pdf', bbox_inches='tight')

plt.show()

for T in range(0,Tmax+1,step):
    print(T,'&',round(sum([Queue_online[T,i]**2 for i in range(N) ]),3),'&',round(sum([Queue_greedy_opt[T,i]**2 for i in range(N)]) ,3),'\\' )


for T in range(0,Tmax+1,step):
    
    if T>0:
        print(T,'&',round(Alg_work_out[T]/T,3),'&',round(Opt_work_out_restricted[T]/T,3),'&',round(Opt_work_out[T]/T,3),'\\')
    
print('% difference of opt work',np.mean([abs( Opt_work_out[T]-Alg_work_out[T] )/Opt_work_out[T] for T in X])*100)

print('% difference of (1-eps)', np.mean([abs( Opt_work_out_restricted[T]-Alg_work_out[T] )/Alg_work_out[T] for T in X])*100)

 
    