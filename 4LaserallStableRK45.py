import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import xlwt
import time

plt.close('all')

def samelength(lists):
    #Get all lists into the same length#  
    minlen = len(min(lists,key=len))
    for i in lists:
        while len(i) != minlen:
            i.pop()       
    X1list = lists[0]
    X2list = lists[1]
    X3list = lists[2]
    X4list = lists[3]
    G1list = lists[4]
    G2list = lists[5]
    G3list = lists[6]
    G4list = lists[7]
    PD12list = lists[8]
    PD23list = lists[9]
    PD34list = lists[10]
    PD41list = lists[11]
    time = lists[12]
    h = lists[13]

    return lists

def ERK45(X_A, X_B, X_C, G_A, PD_AB, PD_CA, h, args, coeffs):

    tao_c, tao_f, a, p, k_ba, k_ca, delta_w, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs
    
    #X1#
    k1 = h*(1/tao_c)*((G_A - a)*X_A - k_ba*np.cos(PD_AB)*X_B - k_ca*np.cos(PD_CA)*X_C) #h * dE1/dt
    X_Aa = X_A + a1*k1
    k2 = h*(1/tao_c)*((G_A - a)*X_Aa - k_ba*np.cos(PD_AB)*X_B  - k_ca*np.cos(PD_CA)*X_C)
    X_Ab = X_A + b1*k1 + b2*k2
    k3 = h*(1/tao_c)*((G_A - a)*X_Ab - k_ba*np.cos(PD_AB)*X_B  - k_ca*np.cos(PD_CA)*X_C)
    X_Ac = X_A + c1*k1 + c2*k2 + c3*k3
    k4 = h*(1/tao_c)*((G_A - a)*X_Ac - k_ba*np.cos(PD_AB)*X_B  - k_ca*np.cos(PD_CA)*X_C)
    X_Ad = X_A + d1*k1 + d2*k2 + d3*k3 + d4*k4
    k5 = h*(1/tao_c)*((G_A - a)*X_Ad - k_ba*np.cos(PD_AB)*X_B  - k_ca*np.cos(PD_CA)*X_C)
    X_Ae = X_A + e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5
    k6 = h*(1/tao_c)*((G_A - a)*X_Ae - k_ba*np.cos(PD_AB)*X_B  - k_ca*np.cos(PD_CA)*X_C)
    RK4 = X_A + fo1*k1 + fo3*k3 + fo4*k4 + fo5*k5
    RK5 = X_A + fi1*k1 + fi3*k3 + fi4*k4 + fi5*k5 + fi6*k6
    diff = abs(RK4-RK5)
    if (diff == 0.):
        return RK4, RK5, float('nan'), diff
    else:
        s = (esp[0]/(2*diff))**(0.25)
        hnew = h*s
        if (hnew < hmax) and (hnew > hmin):
            return RK4, RK5, hnew, diff
        else:
            return RK4, RK5, float('nan'), diff

def GRK45(X_A, G_A, h, args, coeffs):

    tao_c, tao_f, a, p, k_ba, k_ca, delta_w, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs

    #G1#
    k1 = h*((p/tao_f) - (G_A/tao_f)*(1 + X_A**2)) #h * dG1/dt
    G_Aa = G_A + a1*k1
    k2 = h*((p/tao_f) - (G_Aa/tao_f)* (1 + X_A**2))
    G_Ab = G_A + b1*k1 + b2*k2
    k3 = h*((p/tao_f) - (G_Ab/tao_f)* (1 + X_A**2))
    G_Ac = G_A + c1*k1 + c2*k2 + c3*k3
    k4 = h*((p/tao_f) - (G_Ac/tao_f)* (1 + X_A**2))
    G_Ad = G_A + d1*k1 + d2*k2 + d3*k3 + d4*k4
    k5 = h*((p/tao_f) - (G_Ad/tao_f)* (1 + X_A**2))
    G_Ae = G_A + e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5
    k6 = h*((p/tao_f) - (G_Ae/tao_f)* (1 + X_A**2))
    RK4 = G_A + fo1*k1 + fo3*k3 + fo4*k4 + fo5*k5
    RK5 = G_A + fi1*k1 + fi3*k3 + fi4*k4 + fi5*k5 + fi6*k6
    diff = abs(RK4-RK5)
    if (diff == 0.):
        return RK4, RK5, float('nan'), diff
    else:
        s = (esp[1]/(2*diff))**(0.25)
        hnew = h*s
        if (hnew < hmax) and (hnew > hmin):
            return RK4, RK5, hnew, diff
        else:
            return RK4, RK5, float('nan'), diff
    
def DPRK45(X_A, X_B, DP_AB, h, args, coeffs):
    tao_c, tao_f, a, p, k_ba, k_ca, delta_w_AB, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs

    #deltaphi#
    k1 = h*((k_ba/tao_c)*(X_A/X_B + X_B/X_A)*np.sin(DP_AB) + delta_w_AB)
    DP_ABa = DP_AB + a1*k1
    k2 = h*((k_ba/tao_c)*(X_A/X_B + X_B/X_A)*np.sin(DP_ABa) + delta_w_AB)
    DP_ABb = DP_AB + b1*k1 + b2*k2
    k3 = h*((k_ba/tao_c)*(X_A/X_B + X_B/X_A)*np.sin(DP_ABb) + delta_w_AB)
    DP_ABc = DP_AB + c1*k1 + c2*k2 + c3*k3
    k4 = h*((k_ba/tao_c)*(X_A/X_B + X_B/X_A)*np.sin(DP_ABc) + delta_w_AB)
    DP_ABd = DP_AB + d1*k1 + d2*k2 + d3*k3 + d4*k4
    k5 = h*((k_ba/tao_c)*(X_A/X_B + X_B/X_A)*np.sin(DP_ABd) + delta_w_AB)
    DP_ABe = DP_AB + e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5
    k6 = h*((k_ba/tao_c)*(X_A/X_B + X_B/X_A)*np.sin(DP_ABe) + delta_w_AB)
    RK4 = DP_AB + fo1*k1 + fo3*k3 + fo4*k4 + fo5*k5
    RK5 = DP_AB + fi1*k1 + fi3*k3 + fi4*k4 + fi5*k5 + fi6*k6
    diff = abs(RK4-RK5)
    if (diff == 0.):
        return RK4, RK5, float('nan'), diff
    else:
        s = (esp[2]/(2*diff))**(0.25)
        hnew = h*s
        if (hnew < hmax) and (hnew > hmin):
            return RK4, RK5, hnew, diff
        else:
            return RK4, RK5, float('nan'), diff

def timeEvolutionRK45(X0, G0, phi0, h0, timestop, args, stabCond, ssCheck):
    """
    Plots evolution of mutually coupled lasers
    using RK45 method with variable timesteps and steady state detection
    see Mutual Injection-Locking and Coherent Combining Chen et. al equation(16)
    
    input:
        E0: Initial E fields
        phi0: Initial phases
        h: Time step (ns)
        timestop: End of simulation
        args: arguments
        
    """
    hcurrent = h0
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, \
        fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 \
        = 1./4, 3./32, 9./32, 1932./2197., -7200./2197, 7296./2197, \
          439./216, -8, 3680./513, -845./4104, \
          -8./27, 2, -3544./2565, 1859./4104, -11./40., \
          25./216, 0., 1408./2565, 2197./4104, -1./5, \
          16./135, 0., 6656./12825, 28561./56430, -90./50, 2./55

    coeffs = a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, \
        fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6
    
    X1list = []
    X2list = []
    X3list = []
    X4list = []
    G1list = []
    G2list = []
    G3list = []
    G4list = []
    DP12list = []
    DP23list = []
    DP34list = []
    DP41list = []
    timelist = []
    hlist = []
    
    X1list.append(X0[0][0])
    X2list.append(X0[1][0])
    X3list.append(X0[2][0])
    X4list.append(X0[3][0])
    G1list.append(G0[0][0])
    G2list.append(G0[1][0])
    G3list.append(G0[2][0])
    G4list.append(G0[3][0])    
    DP12list.append(phi0[0][0])
    DP23list.append(phi0[1][0])
    DP34list.append(phi0[2][0])
    DP41list.append(phi0[3][0])
    timelist.append(0)
    hlist.append(0)

    tao_c, tao_f, a, p, k_array, delta_w_array, esp, hmin, hmax = args
    
    args1 = tao_c, tao_f, a, p, k_array[1][0], k_array[3][0], delta_w_array[0][1], esp, hmin, hmax
    args2 = tao_c, tao_f, a, p, k_array[2][1], k_array[0][1], delta_w_array[1][2], esp, hmin, hmax
    args3 = tao_c, tao_f, a, p, k_array[3][2], k_array[1][2], delta_w_array[2][3], esp, hmin, hmax
    args4 = tao_c, tao_f, a, p, k_array[0][3], k_array[2][3], delta_w_array[3][0], esp, hmin, hmax
    
    timecurrent = 0
    X1steadyStateBreak = 0
    DPsteadyStateBreak = 0
                   
    while timecurrent < timestop:
        #Get current values from last entry in lists
        x1, x2, x3, x4, g1, g2, g3, g4, dp12, dp23, dp34, dp41 = X1list[-1], X2list[-1], X3list[-1], X4list[-1], G1list[-1], G2list[-1], G3list[-1], G4list[-1], \
                                                                 DP12list[-1], DP23list[-1], DP34list[-1], DP41list[-1]
        #Determine min value of h#
        X1RK4, X1RK5, X1hnew, X1diff = ERK45(x1, x2, x4, g1, dp12, -dp41, hcurrent, args1, coeffs)
        G1RK4, G1RK5, G1hnew, G1diff = GRK45(x1, g1, hcurrent, args1, coeffs)
        X2RK4, X2RK5, X2hnew, X2diff = ERK45(x2, x3, x1, g2, dp23, -dp12, hcurrent, args2, coeffs)
        G2RK4, G2RK5, G2hnew, G2diff = GRK45(x2, g2, hcurrent, args2, coeffs)
        X3RK4, X3RK5, X3hnew, X3diff = ERK45(x3, x4, x2, g3, dp34, -dp23, hcurrent, args3, coeffs)
        G3RK4, G3RK5, G3hnew, G3diff = GRK45(x3, g3, hcurrent, args3, coeffs)
        X4RK4, X4RK5, X4hnew, X4diff = ERK45(x4, x1, x3, g4, dp41, -dp34, hcurrent, args4, coeffs)
        G4RK4, G4RK5, G4hnew, G4diff = GRK45(x4, g4, hcurrent, args4, coeffs)
        DP12RK4, DP12RK5, DP12hnew, DP12diff = DPRK45(x1, x2, dp12, hcurrent, args1, coeffs)
        DP23RK4, DP23RK5, DP23hnew, DP23diff = DPRK45(x2, x3, dp23, hcurrent, args2, coeffs)
        DP34RK4, DP34RK5, DP34hnew, DP34diff = DPRK45(x3, x4, dp34, hcurrent, args3, coeffs)
        DP41RK4, DP41RK5, DP41hnew, DP41diff = DPRK45(x4, x1, dp41, hcurrent, args4, coeffs)

        hnewlist = [X1hnew, G1hnew, X2hnew, G2hnew, X3hnew, G3hnew, X4hnew, G4hnew, DP12hnew, DP23hnew, DP34hnew, DP41hnew]
        difflist = [X1diff, G1diff, X2diff, G2diff, X3diff, G3diff, X4diff, G4diff, DP12diff, DP23diff, DP34diff, DP41diff]
        hcurrent = np.nanmin(hnewlist)
##        print hcurrent, hnewlist, difflist
        if np.isnan(hcurrent):
            hcurrent = hmax
            hlist.append(hmax)
        else:
            hlist.append(hcurrent)
##        print hcurrent, hnewlist
##        print difflist
        
        #Time#
        timecurrent += hcurrent
        timelist.append(timecurrent)
        
        #Calculates values for t+h
        X1next, X1RK5, X1hnew, X1diff = ERK45(x1, x2, x4, g1, dp12, -dp41, hcurrent, args1, coeffs)
        if not math.isinf(X1next):
            X1list.append(X1next)
            x1 = X1next
        else:          
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        G1next, G1RK5, G1hnew, G1diff = GRK45(x1, g1, hcurrent, args1, coeffs)
        if not math.isinf(G1next):
            G1list.append(G1next)
            g1 = G1next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)
        
        X2next, X2RK5, X2hnew, X2diff = ERK45(x2, x3, x1, g2, dp23, -dp12, hcurrent, args2, coeffs)
        if not math.isinf(X2next):
            X2list.append(X2next)
            x2 = X2next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        G2next, G2RK5, G2hnew, G2diff = GRK45(x2, g2, hcurrent, args2, coeffs)
        if not math.isinf(G2next):
            G2list.append(G2next)
            g2 = G2next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        X3next, X3RK5, X3hnew, X3diff = ERK45(x3, x4, x2, g3, dp34, -dp23, hcurrent, args3, coeffs)
        if not math.isinf(X3next):
            X3list.append(X3next)
            x3 = X3next
        else:          
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        G3next, G3RK5, G3hnew, G3diff = GRK45(x3, g3, hcurrent, args3, coeffs)
        if not math.isinf(G3next):
            G3list.append(G3next)
            g3 = G3next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)
        
        X4next, X4RK5, X4hnew, X4diff = ERK45(x4, x1, x3, g4, dp41, -dp34, hcurrent, args4, coeffs)
        if not math.isinf(X4next):
            X4list.append(X4next)
            x4 = X4next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        G4next, G4RK5, G4hnew, G4diff = GRK45(x4, g4, hcurrent, args4, coeffs)
        if not math.isinf(G4next):
            G4list.append(G4next)
            g4 = G4next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        DP12next, DP12RK5, DP12hnew, DP12diff = DPRK45(x1, x2, dp12, hcurrent, args1, coeffs)
        if not math.isnan(DP12next):
            DP12list.append(divmod(DP12next,(2*np.pi))[1])
            dp12 = DP12next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        DP23next, DP23RK5, DP23hnew, DP23diff = DPRK45(x2, x3, dp23, hcurrent, args2, coeffs)
        if not math.isnan(DP23next):
            DP23list.append(divmod(DP23next,(2*np.pi))[1])
            dp23 = DP23next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        DP34next, DP34RK5, DP34hnew, DP34diff = DPRK45(x3, x4, dp34, hcurrent, args3, coeffs)
        if not math.isnan(DP34next):
            DP34list.append(divmod(DP34next,(2*np.pi))[1])
            dp34 = DP34next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        DP41next, DP41RK5, DP41hnew, DP41diff = DPRK45(x4, x1, dp41, hcurrent, args4, coeffs)
        if not math.isnan(DP41next):
            DP41list.append(divmod(DP41next,(2*np.pi))[1])
            dp41 = DP41next
        else:
            lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
            return samelength(lists)

        #############Check for steady state to end sim early############
        X1len = len(X1list)
        DP12len = len(DP12list)
        if(X1len > 1000) and (timecurrent > tao_c*1e4):
            X1_ss_para = np.std(10*np.log10(scipy.array(X1list[-1000:-1]))) #Check for steady state to end sim prematurely
            if X1_ss_para < ssCheck[0]:
                DP_ss_para = np.std(scipy.array(DP12list[-1000:-1])) #Check for steady state to end sim prematurely
##                print X1_ss_para, DP_ss_para
                if DP_ss_para < ssCheck[1]:
                    lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
                    return samelength(lists)

    ############Output lists at end of timestop##############
    lists = [X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, DP12list, DP23list, DP34list, DP41list, timelist, hlist]
    return samelength(lists)

def singlelaser(k, delta_w):
    args = tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax 
    
    X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, PD12list, PD23list, PD34list, PD41list, timelist, hlist \
            = timeEvolutionRK45(E0, G0, phi0, h0, timestop, args, stabCond, ssCheck)

    for i in range(2):
        X1list.pop()
        X2list.pop()
        X3list.pop()
        X4list.pop()
        G1list.pop()
        G2list.pop()
        G3list.pop()
        G4list.pop()
        PD12list.pop()
        PD23list.pop()
        PD34list.pop()
        PD41list.pop()
        timelist.pop()
        hlist.pop()

    stable = np.std(scipy.array(PD12list[-50:-1]))
    stable1 = np.std(scipy.array(X1list[-50:-1]))
    stable2 = np.std(scipy.array(G1list[-50:-1]))

    print stable, stable1, stable2
    
    if (stable < stabCond[2]) and (stable1 < stabCond[0]) and (stable2 < stabCond[1]):
        print 'stable'
    else:
        print 'unstable'

    print 'Single Laser E is %f' %abs(X1list[-1])
    print 'Single Laser G is %f' %abs(G1list[-1])
    
    if plot == 'yes':
    ##        plt.figure(1)
            plt.plot(timelist, X1list,label='Single laser')
            plt.legend()
    ##        plt.xlabel('Time (ps)')
    ##        plt.ylabel('$E_{single laser}$')
    ##        plt.title('Time evolution of E field amplitude')
                
    ##        plt.figure(2)
    ##        plt.plot(timelist, G1list,label='G1')
    ##        plt.plot(timelist, G2list,label='G2')
    ##        plt.xlabel('Time (ps)')
    ##        plt.ylabel('Gain')
    ##        plt.title('Time evolution of laser gain')
    ##        
    ##        plt.figure(3)
    ##        plt.plot(timelist,scipy.array(deltaphilist)/(np.pi))
    ##        plt.xlabel('Time (fs)')
    ##        plt.ylabel('phase difference ($\Delta\phi/\pi$)')
    ##        plt.title('Time evolution of phase difference $\Delta\phi$')

    else:
        return X1list[-1], G1list[-1]

def evolution(k_array, delta_w_array, stabCond, ssCheck):
    args = tao_c, tao_f, a, p, k_array, delta_w_array, esp, hmin, hmax 
    print k_array, delta_w_array
    k = k_array[0][0]
    delta_w = delta_w_array[0][0]
    if k != 0.0:
        print delta_w*tao_c/(2*k)
    X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, PD12list, PD23list, PD34list, PD41list, timelist, hlist = \
            timeEvolutionRK45(E0, G0, phi0, h0, timestop, args, stabCond, ssCheck)

    for i in range(2):
        X1list.pop()
        X2list.pop()
        X3list.pop()
        X4list.pop()
        G1list.pop()
        G2list.pop()
        G3list.pop()
        G4list.pop()
        PD12list.pop()
        PD23list.pop()
        PD34list.pop()
        PD41list.pop()
        timelist.pop()
        hlist.pop()

    stable = np.std(scipy.array(PD12list[-50:-1]))
    stable1 = np.std(scipy.array(X1list[-50:-1]))
    stable2 = np.std(scipy.array(G1list[-50:-1]))

    print stable, stable1, stable2
    
    if (stable < stabCond[2]) and (stable1 < stabCond[0]) and (stable2 < stabCond[1]):
        print 'stable'
    else:
        print 'unstable'
    
     #Plot comparison plot
##    plt.figure(1)
##    plt.plot(timelist, X1list,label='X1 $\kappa$ = 0.1, $\Delta\omega$ = 3000 GHz')
##    plt.plot(timelist, X2list,label='X2 $\kappa$ = 0.1, $\Delta\omega$ = 3000 GHz')
##    plt.xlabel('Time (ps)')
##    plt.ylabel('E field')
##    plt.title('Time evolution of E field amplitude')

    #Normalise X1 magnitude to that of single laser and convert to dB
    X1norm = 10* np.log10(scipy.array(X1list)/Esl)
    X2norm = 10* np.log10(scipy.array(X2list)/Esl)
    X3norm = 10* np.log10(scipy.array(X3list)/Esl)
    X4norm = 10* np.log10(scipy.array(X4list)/Esl)

    label =  ' $\kappa$ = %1.1f, $\Delta\omega$ = %1.0f' %(k, delta_w*1e3)
    
    plt.figure(1)
    plt.plot(timelist, X1norm,label='X1')
    plt.plot(timelist, X2norm,label='X2')
    plt.plot(timelist, X3norm,label='X3')
    plt.plot(timelist, X4norm,label='X4')
    plt.legend()
    plt.xlabel('Time (ps)')
    plt.ylabel('$E/E_{single laser}$ (dB)')
    plt.title('Time evolution of E field amplitude' + label)
        
    plt.figure(2)
    plt.plot(timelist, G1list,label='G1')
    plt.plot(timelist, G2list,label='G2')
    plt.plot(timelist, G3list,label='G3')
    plt.plot(timelist, G4list,label='G4')
    plt.xlabel('Time (ps)')
    plt.ylabel('Gain')
    plt.title('Time evolution of laser gain' + label)
    
    plt.figure(3)
    plt.plot(timelist,scipy.array(PD12list)/(np.pi), label='PD12')
    plt.plot(timelist,scipy.array(PD23list)/(np.pi), label='PD23')
    plt.plot(timelist,scipy.array(PD34list)/(np.pi), label='PD34')
    plt.plot(timelist,scipy.array(PD41list)/(np.pi), label='PD41')
    plt.xlabel('Time (ps)')
    plt.ylabel('Phase difference ($\Delta\phi/\pi$)')
    plt.title('Time evolution of phase difference' + label)

    plt.figure(4)
    plt.plot(timelist,scipy.array(hlist))
    plt.xlabel('Time (ps)')
    plt.ylabel('Time step (ps)')
    plt.title('Time evolution of timestep' + label)

    plt.figure(5)
    plt.plot(timelist, (scipy.array(X1list)/Esl),label='X1')
    plt.plot(timelist, (scipy.array(X2list)/Esl),label='X2')
    plt.plot(timelist, (scipy.array(X3list)/Esl),label='X3')
    plt.plot(timelist, (scipy.array(X4list)/Esl),label='X4')
    plt.legend()
    plt.xlabel('Time (ps)')
    plt.ylabel('$E/E_{single laser}$')
    plt.title('Time evolution of E field amplitude' + label)

##    Ess = X1list[-1]
##    Eratio = Ess/Esl 
##    print 'Steady state E field is %f x E_sl' %(Eratio)
    
    return X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, PD12list, PD23list, PD34list, PD41list, timelist, hlist 
    
def plot(path, knpoints, dwspacing, stabCond, injRatioRange, ssCheck):
    '''
    Plots value of delta_phi and Ess/Esl for different values of k and delta_w
    '''
    starttime = time.time()
    wb = xlwt.Workbook()
    ws = wb.add_sheet('phase difference')
    ws.write(0,0,'inj ratio (dB)')
    ws.write(0,1,'frequency detuning (GHz)')
    ws.write(0,2,'phase difference (1/pi)')
    ws.write(0,3,'E/E_sl (dB)')
    ws.write(0,4,'G/G_sl')
    currentrow = 1

    #Lists containing only stable values
    deltawstablelist=[]
    kdbstablelist = []
    philist=[]
    Elist = []
    Glist = []
   
    #Lists containing all values
    kdblist = []
    dwnumberlist = [] #List of number of dw points for each value in kdblist

    for j in np.linspace(injRatioRange[0], injRatioRange[1], knpoints):
        kdblist.append(round(j,4))
        dwnumber = 1 #Variable to store number of dw points for kdb = j
        dwlimit = 2*10**(scipy.array(round(j,4))/10)/tao_c 
        for i in np.arange(-dwlimit, dwlimit, dwspacing): #(THz) Must be same as value in line 305
            dwnumber +=1
        dwnumberlist.append(dwnumber) #Stores number of dw points into list
    dwtotal = scipy.array(dwnumberlist).sum() #Total number of dw points

    #Start Loop to calculate values for different k and dw values
    for x in range(len(kdblist)):
        kdb = kdblist[x]
        k = 10**(kdb/10)
        print '%f: k = %f(db), k = %f' % (x, kdb, k)

        k_array = np.ndarray(shape=(4,4))
        for i in range(len(k_array)):
            for j in range(len(k_array[i])):
                           k_array[i][j] = k
        
        #Limit dw values to within stable region#
        dwlimit = 2*10**(scipy.array(kdb)/10)/tao_c
        deltawlist = []
##        deltawlistextra = []
        deltawlist.append(0.0)
        for i in np.arange(-dwlimit, dwlimit, dwspacing): #(THz)
            deltawlist.append(i)

##        for d in np.arange(-dwlimit,dwlimit, dwspacing/2):
##            if not (d in deltawlist):
##                deltawlistextra.append(d)
        
        #Remaining Time Calculation#
        dwcurrent = dwnumberlist[x]
        dwleft = scipy.array(dwnumberlist)[x:].sum()
        dwpassed = dwtotal - dwleft 
        if (x>0):
            timeelapsed = time.time() - starttime
            print 'Time elapsed: %f min' %(timeelapsed/60.)
            timeleft = (dwleft)*(timeelapsed/dwpassed)
            print 'Time left: %f min' %(timeleft/60.)

        for y in range(len(deltawlist)):
            delta_w = deltawlist[y]
            delta_w_array = np.ndarray(shape=(4,4))
            for i in range(len(delta_w_array)):
                for j in range(len(delta_w_array[i])):
                    delta_w_array[i][j] = delta_w
                               
            args = tao_c, tao_f, a, p, k_array, delta_w_array, esp, hmin, hmax

            X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, PD12list, PD23list, PD34list, PD41list, timelist, hlist = \
                    timeEvolutionRK45(E0, G0, phi0, h0, timestop, args, stabCond, ssCheck)

            for i in range(2):
                X1list.pop()
                X2list.pop()
                X3list.pop()
                X4list.pop()
                G1list.pop()
                G2list.pop()
                G3list.pop()
                G4list.pop()
                PD12list.pop()
                PD23list.pop()
                PD34list.pop()
                PD41list.pop()
                timelist.pop()
                hlist.pop()
            
            stable = np.std(scipy.array(PD12list[-50:-1]))
            stable1 = np.std(scipy.array(X1list[-50:-1]))
            stable2 = np.std(scipy.array(G1list[-50:-1]))

##            print stable, stable1, stable2
            
            if (stable < stabCond[2]) and (stable1 < stabCond[0]) and (stable2 < stabCond[1]):
                Ess = abs(X1list[-1])
                Eratio = Ess/Esl
                Gss = abs(G1list[-1])
                Gratio = Gss/Gsl
                
                deltawstablelist.append(delta_w*1e3) #Add stable delta_w value in GHz
                kdbstablelist.append(kdb) #Stable inj ratio value in dB
                philist.append(deltaphilist[-1]) #Stable state phase difference in radians
                Elist.append(10*np.log10(Eratio)) #Stable state E_ss/E_singlelaser in dB 
                Glist.append(Gratio)
                
                ws.write(currentrow, 0, kdb)
                ws.write(currentrow, 1, delta_w)
                ws.write(currentrow, 2, deltaphilist[-1])
                ws.write(currentrow, 3, 10*np.log10(Eratio))
                ws.write(currentrow, 4, Gratio)
                currentrow += 1
                
            else:
                ws.write(currentrow, 0, kdb)
                ws.write(currentrow, 1, delta_w)
                ws.write(currentrow, 2, 'unstable')
                ws.write(currentrow, 3, 'unstable')
                ws.write(currentrow, 4, 'unstable')
                currentrow += 1
    
        wb.save(path)

    plt.figure(1)
    phicolour = scipy.array(philist)/(np.pi)
    cm = plt.cm.get_cmap()
    sc = plt.scatter(kdbstablelist, deltawstablelist, c=phicolour, cmap=cm, edgecolors='none')
    plt.colorbar(sc)
    plt.xlabel('Injection Ratio (dB)')
    plt.ylabel('Frequency Detuning (GHz)')
    plt.title('Phase Difference $\Delta\phi/\pi$')
    deltawlimit = 2*1e3*(10**(scipy.array(kdbstablelist)/10))/tao_c #Stability limit in GHz
    plt.plot(kdbstablelist,deltawlimit)
    deltawlimitneg = -2*1e3*(10**(scipy.array(kdbstablelist)/10))/tao_c #Stability limit in GHz
    plt.plot(kdbstablelist,deltawlimitneg)
    
    plt.figure(2)
    Ecolour = scipy.array(Elist)
    cm = plt.cm.get_cmap()
    sc = plt.scatter(kdbstablelist, deltawstablelist, c=Ecolour, cmap=cm, edgecolors='none')
    plt.colorbar(sc)
    plt.xlabel('Injection Ratio (dB)')
    plt.ylabel('Frequency Detuning (GHz)')
    plt.title('$E/E_{sl}$ (dB)')
    deltawlimit = 2*1e3*(10**(scipy.array(kdbstablelist)/10))/tao_c #Stability limit in GHz
    plt.plot(kdbstablelist,deltawlimit)
    deltawlimitneg = -2*1e3*(10**(scipy.array(kdbstablelist)/10))/tao_c #Stability limit in GHz
    plt.plot(kdbstablelist,deltawlimitneg)

    plt.figure(3)
    Gcolour = scipy.array(Glist)
    cm = plt.cm.get_cmap()
    sc = plt.scatter(kdbstablelist, deltawstablelist, c=Gcolour, cmap=cm, edgecolors='none')
    plt.colorbar(sc)
    plt.xlabel('Injection Ratio (dB)')
    plt.ylabel('Frequency Detuning (GHz)')
    plt.title('$G/G_{sl}$ (dB)')
    deltawlimit = 2*1e3*(10**(scipy.array(kdbstablelist)/10))/tao_c #Stability limit in GHz
    plt.plot(kdbstablelist,deltawlimit)
    deltawlimitneg = -2*1e3*(10**(scipy.array(kdbstablelist)/10))/tao_c #Stability limit in GHz
    plt.plot(kdbstablelist,deltawlimitneg)

#######################Main###########################
L = 5*1e6 #Optical length of one round trip (pm)
c = 3e8 #Speed of light (pm/ps)
tao_c = 2*L/c #Round trip time (ps)
tao_f = 1e3 #Fluorescence time (ps)
a = 0.1 #cavity loss coefficient
p = 2. #pumping coefficient

k_sl = np.ndarray(shape=(4,4))
for i in range(len(k_sl)):
    for j in range(len(k_sl[i])):
        k_sl[i][j] = 0.
                           
delta_w_sl = np.ndarray(shape=(4,4))
for i in range(len(delta_w_sl)):
    for j in range(len(delta_w_sl[i])):
        delta_w_sl[i][j] = 0.

k_12, k_23, k_34, k_41 = 0.01, 0.01, 0.01, 0.01
k_array = np.array([[0., k_12, 0.0, k_41],[k_12, 0., k_23, 0.],[0.0, k_23, 0., k_34],[k_41, 0.0, k_34, 0.]])

dw_12, dw_23, dw_34, dw_41 = 0.1, 0.1, 0.1, 0.1 
delta_w_array = np.array([[0., dw_12, 0.0, -dw_41],[-dw_12, 0., dw_23, 0.0],[0.0, -dw_23, 0., dw_34],[dw_41, 0., -dw_34, 0.]])

#############################RK4 parameters############################
timestop = 1000 #(ps)
h0 = 5e-3 #(ps) Starting timestep
E0 = [[1.],[1.],[1.],[1.]]
G0 = [[2.],[2.],[2.],[2.]]
phi0 = [[0.0],[0.0],[0.0],[0.0]]
esp = [0.05, 1e-3, 0.005] #Error allowance parameter for [X, G, DP]
hmin = 1e-4 #Min timestep
hmax = 1e-2 #Max timestep
stabCond = [0.3, 1e-3, 0.03] #Stability conditions for [X,G,DP]
ssCheck = [0.2, 1e-4] #Steady state detector paramters for [X, DP]
injRatioRange = [-25, -23] #Injection ratio range(dB)
injRatioNPoints = 30
dwSpacing = 0.05

############################Running Simulations#########################

##print 'Calculating single laser'
##E,G = singlelaser(k_sl,delta_w_sl)
##Esl,Gsl = abs(E), abs(G)
##print

Esl, Gsl = 4.3, 0.1 

print 'Calculating Time evolution'
X1list, X2list, X3list, X4list, G1list, G2list, G3list, G4list, PD12list, PD23list, PD34list, PD41list, timelist, hlist = evolution(k_array, delta_w_array, stabCond, ssCheck) #(k,delta_w(THz))    
plt.show()

##for i in np.linspace(11.58048122,11.58048133,10):
##    evolution(0.2, i)
##    print

##plot('AllstableRK45_13.xls', injRatioNPoints, dwSpacing, stabCond, injRatioRange, ssCheck) #plot(path, knpoints, dwspacing, stabCond, injRatioRange, ssCheck)
##plt.show()

#Plot stable region#
##kdb = np.linspace(-20,-6,100)
##deltaw = 2*10**(scipy.array(kdb)/10)*1e3/tao_c
##plt.plot(kdb,deltaw)
##plt.plot(kdb, -deltaw)
##plt.xlabel('Injection Ratio (dB)')
##plt.ylabel('Frequency Detuning (GHz)')
##plt.title('Mutual Injection stable region')
##plt.show()
