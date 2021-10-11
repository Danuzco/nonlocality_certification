import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt
import os
import itertools

from datetime import datetime
cwd = os.getcwd()


def optimal_value(filename, m, num_of_outcomes, num_of_trials, F, marginals_A, 
                  marginals_B, disp = False, save = False):
    
    """
    This routine optimizes R 
    
    Arguments:
    m: number of settings.
    filename: Name of the file where the experimental data is stored.
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.
    disp: 'True' for displaying the partial results.
    save: 'True' to save the coefficients S, Sax and Sby as numpy tensors in the 'npy_data' folder.
                         
    Returns: 
    Tensors "S", "Sax" and "Sby" containing the coefficients 
    """
    
    foundResults = {}
    SolutionFound = False
    dir1 = ( cwd + '\\npy_data\\' + 's_' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    dir2 = ( cwd + '\\npy_data\\' + 'sax_' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    dir3 = ( cwd + '\\npy_data\\' + 'sby_' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    dir4 = ( cwd + '\\npy_data\\' + 'p_' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    dir5 = ( cwd + '\\npy_data\\' + 'pax_' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    dir6 = ( cwd + '\\npy_data\\' + 'pby_' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    dir7 = ( cwd + '\\npy_data\\' + 'Q_C_Dq_Gap' + 
        filename + datetime.today().strftime('_%Y-%m-%d') )
    
    print('n: number of iterations')
    print(60*'_')
    print(f'    n       R         Q         C         \u0394Q         Gap')
    print(60*'_')
    p, c, _ = load_data(filename, F, m, num_of_outcomes, 
                        marginals_A, marginals_B)
    s0, bounds = initial_guess(m, num_of_outcomes, 
                               marginals_A, marginals_B) # initial guess for the optimizer
    n, target = 1, -1
    nlc = NonlinearConstraint( lambda x: -target  
                              + R( x, p, c, m, num_of_outcomes, marginals_A, marginals_B), 
                                      -np.inf, -target ) # constraints

    while n <= num_of_trials:
        res = minimize( R , s0 ,  args=(p, c, m, num_of_outcomes, marginals_A, marginals_B), 
                        method='SLSQP', tol = 1e-20, 
                        options={ 'ftol':1e-20, 'maxiter':5000, 'disp': False}, 
                        bounds=bounds, constraints=[nlc])
        s0 = (res.x + s0)/2  # average between the target guess 's0' and the current solution
                             # R = -res.fun. Thus, 'res.fun < target' => 'R > -target'
        if  res.fun < target:
            SolutionFound = True
            solution = res # keeps the solution that satisfies the constraints
            target = res.fun  # updates the target to the best R value found so far
            nlc = NonlinearConstraint( lambda x: -target  
                                      + R( x, p, c, m, num_of_outcomes, marginals_A, marginals_B), 
                                      -np.inf, -target ) # update the constraints
            
            Q, C, Dq, gap = results( solution.x, p, c, m, 
                                    num_of_outcomes, marginals_A, marginals_B)
            s, sax, sby = vector_to_tensor(solution.x, p, m, num_of_outcomes, marginals_A, marginals_B)
            # save solutions at the 'npy_data' folder
            pabxy, pax, pby = p
            if save:
                np.save(dir1, s)
                np.save(dir2, sax)
                np.save(dir3, sby)
                np.save(dir4, pabxy)
                np.save(dir5, pax)
                np.save(dir6, pby)
                np.save(dir7, np.array([Q, C, Dq, gap]) )

            # display results
            if disp:
                print( "%5d    %4.5f   %4.5f   %4.5f    %4.5f    %4.5f" 
                      %(n, -solution.fun, Q, C, Dq, gap) )    
               
        n += 1
    
    if SolutionFound:
        foundResults['Coefficients'] = (s, sax, sby)
        foundResults['values'] = (Q, C, Dq, gap)
        return foundResults
    else:
        print("\nNo solution has been found ...")
    

    
def probs_local_hidden_model(num_of_outcomes , m):
    
    outcomes = [1] + [0 for i in range(num_of_outcomes-1)]
    rows = ( list(i) for i in set( itertools.permutations(outcomes)) )
    
    for i in itertools.product(rows, repeat = m):
        yield np.array(i).T 

        
def lhv_value(s, m, num_of_outcomes ):
    """
    Inputs:
    s: tuple containing the tensors (S, Sax, Sby)
    num_of_outcomes: number of outcomes
    m: number of settings
    
    Returns: 
    local bound for a 2-party scenario, with "m" settings and "num_of_outcomes" possible outcomes
    """
    s, sax, sby = s
    Cmax = -1e10
    for pax in probs_local_hidden_model(num_of_outcomes , m):
        for pby in probs_local_hidden_model(num_of_outcomes , m):
            C =  ( np.sum(sax*pax) + np.sum(sby*pby)  
                  + np.sum( s*np.einsum('ax,by->abxy', pax,pby) ) )
            if C > Cmax:
                Cmax = C
    return Cmax


def quantum_max_violation(s, p):
    """
    Inputs:
    s: tuple containing the tensors (S, Sax, Sby)
    p: tuple containing the tensors (p, pax, pby)
    
    Returns:
    Quantum Value
    """
    s, sax, sby = s
    p, pax, pby = p
    return np.sum(sax*pax) + np.sum(sby*pby) + np.sum(s*p)


def error_quantum_value( s, c, m, num_of_outcomes):
    """
    This function computes the experimental error of the quantum value "Q" from the countings.
    
    Arguments:
    s: tuple containing the tensors (S, Sax, Sby)
    c: a numpy ndarray c[a,b,x,y], containing the experimental countings c(ab|xy)
    m: number of settings 
    num_of_outcomes: number of outcomes
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.

                 
    Returns:
    experimental error
    
    """
    c,cax,cby = c
    s,sax,sby = s
    Delta_Q, Delta_Q1, Delta_Q2 =  0, 0, 0
    for i in itertools.product(range(m), repeat = 2):
        for l in itertools.product(range(num_of_outcomes), repeat = 2):
            x,y = i
            a,b = l
            dQ = ( s[a,b,x,y]*np.sum( c[:, :, x, y] ) 
                  - np.sum(s[:,:,x,y]*c[:,:,x,y] ) )/( np.sum( c[:, :, x, y] )**2 )
            if False:
                dQ1 = ( sax[a,x]*np.sum( cax[:,x] ) 
                       - np.sum(sax[:,x]*cax[:,x] ) )/( np.sum( cax[:,x] )**2 )
                dQ2 = ( sby[b,y]*np.sum( cby[:,y] ) 
                       - np.sum(sby[:,y]*cby[:,y] ) )/( np.sum( cby[:,y] )**2 )
                Delta_Q1 += dQ1**2*cax[a,x]
                Delta_Q2 += dQ2**2*cby[b,y]
            else:
                dQ = dQ + (1/m)*(  sax[a,x]*np.sum(c[:, :, y,x] ) 
                                 - np.sum(sax[:,x]*c[:,:,x,y]) )/( np.sum( c[:, :, x, y] )**2 )
                dQ = dQ + (1/m)*(  sby[b,y]*np.sum(c[:, :, y,x] ) 
                                 - np.sum(sby[:,y]*c[:,:,x,y]) )/( np.sum( c[:, :, x, y] )**2 )
            Delta_Q += dQ**2*c[a,b,x,y]
            
    return np.sqrt( Delta_Q  + Delta_Q1 + Delta_Q2 )



def initial_guess(m, num_of_outcomes, marginals_A, marginals_B):    
    """
    This function returns the initial guess and the bounds to be used for the optimizer
    
    Inputs:
    num_of_outcomes: number of outcomes
    m: number of settings
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.
    
    Returns:
    s0: a numpy array with the initial guess
    bounds: a numpy array with the bounds
    """
    nop = (num_of_outcomes**2)*(m**2) + num_of_outcomes**2*(len(marginals_A) + len(marginals_B))
    lb, ub = -1, 1
    bounds =  tuple( [(lb, ub) for i in range(nop)] )
    s0 = np.array(  [(ub - lb) * np.random.random() + lb   for i in range(nop)] )
    
    return s0, bounds


def R( s, p, c, m, num_of_outcomes, marginals_A, marginals_B):
    """
    inputs:
    s: a numpy 1D array
    p: tuple containing the tensors (p, pax, pby)
    c: a tensor c[a,b,x,y], containing the experimental countings c(ab|xy)
    m: number of settings 
    num_of_outcomes: number of outcomes
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.
    
    Returns:
    - R
    """
    k = (num_of_outcomes)**2*(m)**2
    s = vector_to_tensor(s, p, m, num_of_outcomes, marginals_A, marginals_B)
    Q = quantum_max_violation(s, p)
    C = lhv_value(s, m, num_of_outcomes )
    Dq = error_quantum_value( s, c, m, num_of_outcomes)
    
    return  -(Q - Dq + k)/(C + k)


def vector_to_tensor(s, p, m, num_of_outcomes, marginals_A, marginals_B):
    """
    inputs:
    s: a numpy 1D array
    p: tuple containing the tensors (p, pax, pby)
    m: number of settings 
    num_of_outcomes: number of outcomes
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.
    
    Returns:
    Tensors (S, Sax, Sby)
    """
    o = num_of_outcomes
    n = m**2*o**2
    sax, sby = np.zeros((2,m)), np.zeros((2,m))
    s, s_marginals = np.array( s[:n] ).reshape((o,o,m,m)),  s[n:]
    
    j = 0
    if len(marginals_A) > 0:
        for x in marginals_A:
            for a in range(o):
                sax[a,x] = s_marginals[j]
                j += 1
    if len(marginals_B) > 0:
        for y in marginals_B:
            for b in range(o):
                sby[b,y] = s_marginals[j]
                j += 1
    s[np.nonzero(p[0] == 0)] = np.zeros((s[np.nonzero(p[0] == 0)].shape[0]))
    
    return (s, sax, sby)



def results( s, p, c, m, num_of_outcomes, marginals_A, marginals_B):
    """
    inputs:
    s: a numpy 1D array
    p: tuple containing the tensors (p, pax, pby)
    c: a tensor c[a,b,x,y], containing the experimental countings c(ab|xy)
    m: number of settings 
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.
    
    Returns:
    A tuple: (Q, C, DeltaQ, Q - DeltaQ - C)
    """
    o = num_of_outcomes
    n = m**2*o**2
    sax, sby = np.zeros((2,m)), np.zeros((2,m))
    s, s_marginals = np.array( s[:n] ).reshape((o,o,m,m)),  s[n:]
    
    j = 0
    if len(marginals_A) > 0:
        for x in marginals_A:
            for a in range(o):
                sax[a,x] = s_marginals[j]
                j += 1
    if len(marginals_B) > 0:
        for y in marginals_B:
            for b in range(o):
                sby[b,y] = s_marginals[j]
                j += 1
    
    s[np.nonzero(p[0] == 0)] = np.zeros((s[np.nonzero(p[0] == 0)].shape[0]))
    s = (s, sax, sby)
    Q = quantum_max_violation(s, p)
    C = lhv_value(s, m, o )
    Dq = error_quantum_value( s, c, m, o)
    
    return  Q, C, Dq, (Q - Dq - C)



def load_data(name_of_file, F, m, num_of_outcomes, marginals_A, marginals_B):
    """
    Load the experimental countings saved in "countings_C[C].txt". The file is located in the "experimental_data" folder.
    Each file consists of two columns: 
    1) The labels "abxy" 
    2) the countings c(ab|xy)
    where "a" and "b" are the labels for the outputs of the "x" and "y" settings, respectively.
    Once the countings are loaded, the experimental frecuencies (probabilities) p(ab|xy) are computed.
    
    Inputs:
    filename: Name of the file where the experimental data is stored.
    m: number of settings
    num_of_outcomes: number of outcomes
    marginals_A: a list containing the marginals (expressions like A_i x I ) in the Alice's side that 
                 have to be considered for the calculations.
    marginals_B: a list containing the marginals (expressions like  I x B_i ) in the Bob's side that 
                 have to be considered for the calculations.
    
    Returns:
    p: tuple containing the tensors (p, pax, pby)
    c: a numpy ndarray c[a,b,x,y] containing the countings
       p[a,b,x,y] contains the experimental frecuencies (probabilities).
    measured_marginals: A boolean, which is set to "True" if there are marginal 
                        probabilities in the "countings_C[C].txt" file. 
    
    """
    
    filename = cwd + '\\experimental_data\\' + name_of_file + '.txt'
    
    with open(filename,'r') as f:
        data = f.read()

    data = data.split('\n')
    labels_counts, counts = [] , []
    for i in range(len(data)):
        x,y = tuple( data[i].split()) 
        labels_counts.append(x)
        counts.append(float(y))
    
    countings = dict(zip(labels_counts, counts))
    
    
    shape = (num_of_outcomes,)*2 + (m,)*2 
    p = np.zeros( shape )
    c = np.zeros( shape )
    pax, cax =  np.zeros( (2,2) ), np.zeros( ( 2,2) )
    pby, cby =  np.zeros( (2,2) ), np.zeros( ( 2,2) )
    
    measured_marginals = False
    for l in labels_counts:
        a,b,x,y = l
        if a == '-':
            measured_marginals = True
            cax[int(a),int(x)] = countings[l]
        elif b == '-':   
            measured_marginals = True
            cby[int(b),int(y)] = countings[l]
        else:
            a,b,x,y = list( map( int,l ) )
            c[a,b,x,y] = countings[l]

    for l in labels_counts:
        a,b,x,y = l
        a,b,x,y = list( map( int,l ) )
        p[a,b,x,y] = c[a,b,x,y]/( np.sum( c[:,:,x,y] ) )
    
    p1 = p
    num_of_trials = 10
    output = optimal_settings_KL_divergence(F, p, num_of_outcomes, m, num_of_trials)
    p = get_probs_from_settings(output.x,m)
    
    if measured_marginals:
        for l in labels_counts:
            a,b,x,y = l
            a,b,x,y = list( map( int,l ) )
            pax[int(a), int(x)] = cax[int(a), int(x)]/( np.sum( cax[:,int(x)] ) )
            pby[int(b), int(y)] = cby[int(b), int(y)]/( np.sum( cby[:,int(y)] ) )

    if len(marginals_A) > 0:
        for x in marginals_A:
            for a in range(2):
                pax[a,x] = np.sum(p[a,: ,x ,0])
    if len(marginals_B) > 0:
        for y in marginals_B:
            for b in range(2):
                pby[b,y] = np.sum(p[:,b ,0 ,y])
                 
    return (p, pax, pby) , ( c, cax, cby ), measured_marginals




# Kullback-Leible divergence

def U(x):
    
    beta, gamma, alpha= x[0], x[1], x[2]
    return np.array( [ [ np.exp( -1j*gamma )*np.cos( beta/2 ) , - np.exp( 1j*alpha )*np.sin( beta/2 ) ], 
                    [ np.exp( -1j*alpha )*np.sin( beta/2 ), np.exp( 1j*gamma )*np.cos(beta/2) ] ] )   


def W(theta):
    
    ket0 = np.array([1,0])
    ket1 = np.array([0,1])
    psi = np.cos(theta)*np.kron(ket0,ket0) + np.sin(theta)*np.kron(ket1,ket1)
    
    return np.outer( psi, np.conjugate( psi ) )


def get_probs_from_settings(x, m):
    
    o = 2
    theta, x = x[0],  x[1:]
    rho = W(theta)
    Us = np.array( list( map(U, x.reshape( int(x.shape[0]/3) ,3) ) ) )
    Ua, Ub = Us[:int(Us.shape[0]/2)], Us[int(Us.shape[0]/2):]
    
    pa, pb, p = np.zeros((o,m)), np.zeros((o,m)), np.zeros((o,o,m,m))
    for i in itertools.product(range(m), repeat = 2):
        for l in itertools.product(range(o), repeat = 2):
            x,y = i
            a,b = l
            Pa = np.outer( Ua[x][:,a], np.conjugate( Ua[x][:,a] ) )
            Pb = np.outer( Ub[y][:,b], np.conjugate( Ub[y][:,b] ) )
            p[a,b,x,y] = np.real( np.trace( rho @ np.kron(Pa, Pb) ) )
    
    return  p


def initial_data(m, num_of_parties):
    
    lb , ub = 0, np.pi
    x0 = np.array( [(ub - lb) * np.random.random_sample() 
                    + lb for i in range(3*num_of_parties*m)] )
    
    lb , ub = 0, np.pi
    theta = (ub - lb) * np.random.random_sample() + lb
    x0 = np.insert(x0 , 0, theta )
    
    ### Bounds
    lb , ub = 0, np.pi
    x_bounds = np.array( [(0, np.pi/4)] 
                        + [(lb, ub) for i in range(3*num_of_parties*m)] )
    
    return x0, x_bounds


def kullback_leibe_divergence(x, F, f, num_of_outcomes, m):
    
    """
    f: relative frequencies (original probabilities)
    """
    p = get_probs_from_settings(x, m)
    
    D = 0
    for i in itertools.product(range(num_of_outcomes), repeat = m):
        for j in itertools.product(range(m), repeat = m):
            a,b = i
            x,y = j
            D += F(x,y)*f[a,b,x,y]*(np.log2(f[a,b,x,y])-np.log2(p[a,b,x,y]))
            
    return D



def optimal_settings_KL_divergence(F, probs, num_of_outcomes, 
                                               m, num_of_trials):
    
    """
    This routine minimize the Kullback-Leibe divergence.
    It optimizes over the parameters in the settings.
    """
    
    # one can also use COBYLA in this optimization
    # just comment the next line and uncomment the 
    # line corresponding to COBYLA.
    
    OPTIMIZER = 'SLSQP' 
#     OPTIMIZER = 'COBYLA'

    # initial guess and bounds for the optimizer
    x0, bounds = initial_data(num_of_outcomes, m)
    threshold = 0
    cons = ({'type': 'ineq', 'fun': lambda x: threshold 
             - kullback_leibe_divergence(x, F, probs, num_of_outcomes, m)} )

    res = minimize( kullback_leibe_divergence , x0 ,  
                    args=(F,probs, num_of_outcomes, m), 
                    method=OPTIMIZER, tol = 1e-20, 
                    options={ 'maxiter':250, 'disp': False},
                   constraints=cons
                  )

    previous = res.fun
    result = res
    for i in range(num_of_trials):
        
        x0 = (res.x + x0)/2
        res = minimize( kullback_leibe_divergence , x0 ,  
                    args=(F, probs, num_of_outcomes, m), 
                    method=OPTIMIZER, tol = 1e-20, 
                    options={ 'maxiter':250, 'disp': False},
                    constraints=cons
                  )
        
        # save the best result
        if res.fun < previous:
            previous = res.fun
            result = res
    
    return result



