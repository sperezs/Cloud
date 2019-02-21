# Cloud

This script is used in the numerical experiment in paper https://arxiv.org/abs/1809.02688

The script will compare mainly three different algorithms:

1. The online algorithm described in the article
2. The offline algorithm called "Proportional Greedy" in the article
3. The offline static allocations

The input load is structured in periods (see below), in each period a group of active users will receive loads from a Gamma distribution.

At the end, the script will plot graphs depicting comparison between work done, individual work done by users and queue lengths.

# Installation

No needed

# Usage

In main section set parameters: N, Tmax, step, eps, eta, init_dist, beta, Periods and Active_Periods.

N = number of users to simulate
Tmax = maximum running time
step = the algorithm will sample every "step" units of time
eps = parameter of the online algorithm, usually less than 1/10
eta = parameter of the online algorithm, usually less than 1/3
init_dist = initial allocation for each user, be sure that each coordinate is at least eps/N
Periods = list of times showing endpoints of each period
Active_Periods = list of list of users active in each period


# Example

N = 3

Tmax = 3000000

step = 50000

eps = 0.01

eta = 0.3   


init_dist = [1/N for i in range(N)] 

beta = [0.2,0.3, 0.5]

Periods = [ int(Tmax/6) , int(2*Tmax/6),int(3*Tmax/6), int(4*Tmax/6), int(5*Tmax/6) , Tmax ]

Active_Period = [[1,2],[0,1],[0,2],[1,2],[0,1],[0,2]]



