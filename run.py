import argparse

import numpy as np

from src.animate_scatter import AnimateScatter
from src.whale_optimization import WhaleOptimization

# method to parse command line arguments with argparse module
def parse_cl_args():
    parser = argparse.ArgumentParser()
    
    #adding various agrguments with add_argument()
    parser.add_argument("-nsols", type=int, default=50, dest='nsols', help='number of solutions per generation, default: 50')
    parser.add_argument("-ngens", type=int, default=30, dest='ngens', help='number of generations, default: 20')
    parser.add_argument("-a", type=float, default=2.0, dest='a', help='woa algorithm specific parameter, controls search spread default: 2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='woa algorithm specific parameter, controls spiral, default: 0.5')
    parser.add_argument("-c", type=float, default=None, dest='c', help='absolute solution constraint value, default: None, will use default constraints')
    parser.add_argument("-func", type=str, default='booth', dest='func', help='function to be optimized, default: booth; options: matyas, booth, deckker ,himmel ')
    parser.add_argument("-r", type=float, default=0.25, dest='r', help='resolution of function meshgrid, default: 0.25')
    parser.add_argument("-t", type=float, default=0.1, dest='t', help='animate sleep time, lower values increase animation speed, default: 0.1')
    parser.add_argument("-max", default=False, dest='max', action='store_true', help='enable for maximization, default: False (minimization)')

    args = parser.parse_args()  #parsing arguments with parse args
    return args #returning the parsed arguments

# Defining different benchmark fucntions to test our optimization functions

def deckkers_Aarts (X, Y):
    '''
    contraint = 20
    minimum f(0, 15)= -24771.09
    '''
    x=X**2
    y=Y**2
    z= (x+y)**2
    return (10**5)*x +y -z + (10**(-5))*(z**2)

def booth(X, Y):
    """constraints=10, minimum f(1, 3)=0"""
    return ((X)+(2.0*Y)-7.0)**2+((2.0*X)+(Y)-5.0)**2

def matyas(X, Y):
    """constraints=10, minimum f(0, 0)=0"""
    return (0.26*(X**2+Y**2))-(0.48*X*Y)

def himmelblau(X, Y):
    """constraints=5,
    minimum f(3,2) = 0.0
    minimum f(-3.779310, 3.283186)=0.0
    minimum f(-2.805118, 3.283186)=0.0
    minimum f(3.584458, -1.848126)=0.0
    """
    return ((X**2+Y-11)**2)+(X+Y**2-7)**2


def main():
    args = parse_cl_args() #storing the parsed arguments
    
    #initialising the variables
    nsols = args.nsols
    ngens = args.ngens

    
    #defining the benchmark functions
    funcs = { 'deckker': deckkers_Aarts, 'himmel': himmelblau, 'booth':booth, 'matyas':matyas, }
    func_constraints = {'deckker': 20.0,'himmel':5.0, 'booth':10.0, 'matyas':10.0, }
    
    #check if the function of the argument exists in the list of benchmark functions
    if args.func in funcs:
        func = funcs[args.func]
    else:
        print('Missing supplied function '+args.func+' definition. Ensure function defintion exists or use command line options.')
        return

    #setting the constraint
    if args.c is None:
        if args.func in func_constraints:
            args.c = func_constraints[args.func]
        else:
            print('Missing constraints for supplied function '+args.func+'. Define constraints before use or supply via command line.')
            return

    C = args.c
    constraints = [[-C, C], [-C, C]]

    opt_func = func

    b = args.b
    a = args.a
    a_step = a/ngens

    maximize = args.max
    
    #outpput at each  step  of algo
    opt_alg = WhaleOptimization(opt_func, constraints, nsols, b, a, a_step, maximize) #running WOA on the input fucntion
    solutions = opt_alg.get_solutions()
    colors = [[1.0, 1.0, 1.0] for _ in range(nsols)]
    
    #creating animated scatter plots
    a_scatter = AnimateScatter(constraints[0][0], 
                               constraints[0][1], 
                               constraints[1][0], 
                               constraints[1][1], 
                               solutions, colors, opt_func, args.r, args.t) 
    
    ## updating animated scatter plots for every ngens
    for _ in range(ngens):
        opt_alg.optimize()
        solutions = opt_alg.get_solutions()
        a_scatter.update(solutions)

    opt_alg.print_best_solutions()

if __name__ == '__main__':
    main()
