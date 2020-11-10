import numpy as np

class WhaleOptimization():
    
#initialising the variables
    def __init__(self, opt_func, constraints, nsols, b, a, a_step, maximize=False):
        self._opt_func = opt_func  #which benchmark function is used 
        self._constraints = constraints  #constraints based on benchmark functions
        self._sols = self._init_solutions(nsols) # no.of whales or initial search agents
        self._b = b #constant for defining the shape of the logarithmic spiral
        self._a = a
        self._a_step = a_step
        self._maximize = maximize #maximization or minimization (default)
        self._best_solutions = [] # target 
        
    def get_solutions(self):
        """return solutions"""
        return self._sols
                                                                  
    def optimize(self):
        """solutions randomly encircle, search or attack"""
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0] 
        #include best solution in next generation solutions
        new_sols = [best_sol]
                                                                 
        for s in ranked_sol[1:]:
        	"""
			Note that humpback whales swim around the prey within a shrinking circle and along a 
			spiral-shaped path simultaneously. To model this simultaneous behaviour, we assume that
			 there is a probability of 50% to choose between either the shrinking encircling mechanism
			  or the spiral model to update the position of whales during optimization.
        	"""
        	#checking the probability 
            if np.random.uniform(0.0, 1.0) > 0.5:                                      
                A = self._compute_A()        #getting ready to encircle                                             
                norm_A = np.linalg.norm(A)   

                if norm_A < 1.0:  # the best solution is selected for updating the position of the search agents                                                       
                    new_s = self._encircle(s, best_sol, A)      #encircling                          
                
                else:                                                                     
                    ###select random sol      
                    #A random search agent is chosen when |A vector| >1                                            
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]       
                    new_s = self._search(s, random_sol, A)                                
            else:                                                                         
                new_s = self._attack(s, best_sol)                                         
            new_sols.append(self._constrain_solution(new_s))

        self._sols = np.stack(new_sols)
        self._a -= self._a_step

    def _init_solutions(self, nsols):
        """initialize solutions uniform randomly (stochastic) in space"""
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        sols = np.stack(sols, axis=-1)
        return sols

    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        """find best solution"""
        fitness = self._opt_func(self._sols[:, 0], self._sols[:, 1])
        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]
   
        #best solution is at the front of the list
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])

        return [ s[1] for s in ranked_sol] 

#for displaying best solution at each iteration
    def print_best_solutions(self):
        print('generation best solution history')
        print('([fitness], [solution])')
        for s in self._best_solutions:
            print(s)
        print('\n')
        print('best solution via function -', self._opt_func)
        print('([fitness], [solution])')
        print(sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0])

#updating as per formula of vector A and C
  	#Where components of a vector are linearly decreased from 2 to 0 over the course of iterations and r1 & r2  (r) are random vectors in [0,1].

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=2)
        return (2.0*np.multiply(self._a, r))-self._a

    def _compute_C(self):
        return 2.0*np.random.uniform(0.0, 1.0, size=2)
        
    #updating the position of all whales for next iteration   
    # X(t+1) = Xp(t -  A.D)                                                    
    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)

        # WOA assumes current best soltion (best_sol) is the target prey (Xp)
        # X(t) is position vector of whale or the other solution ((sol))                         
    	#D =|C.Xp(t) - X(t)|
    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        D = np.linalg.norm(np.multiply(C, best_sol)  - sol)
        return D

   
#EXPLOITATION PHASE 
#L=t
#spiral equation 

#X* is the best solution
     #D' = |X*(t) - X(t)|
        #indicates the distance of the i-th whale the prey (best solution obtained so far), 
        #b is a constant for defining the shape of the logarithmic spiral, and t is a random number in [-1,1].
    def _attack(self, sol, best_sol):
        D = np.linalg.norm(best_sol - sol)
        L = np.random.uniform(-1.0, 1.0, size=2)
        return np.multiply(np.multiply(D,np.exp(self._b*L)), np.cos(2.0*np.pi*L))+best_sol
		# X(t+1)=D'*e^(b*t) cos(2pi*t) + X*(t)

 #EXPLORATION PHASE
#Xrand is a random whale or random search agent
    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
            # X(t+1) = Xrand(t)-  A.D                                                    

        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
    	#D=|C*Xrand - X|
    	#X = current solution
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol)    

