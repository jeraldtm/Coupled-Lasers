import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import xlwt
import time
from collections import defaultdict
import numpy.random as random

plt.close('all')

def samelength(lists):
    #Get all lists into the same length#  
    minlen = len(min(lists,key=len))
    for i in lists:
        while len(i) != minlen:
            i.pop()       
    return lists

def ERK45(X_this, X_A, X_B, X_C, X_D, G_this, PD_A, PD_B, PD_C, PD_D, k_a, k_b, k_c, k_d, h, args, coeffs):

    tao_c, tao_f, a, p, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs
    
    #X1#
    inj_term = k_a*np.cos(PD_A)*X_A + k_b*np.cos(PD_B)*X_B + k_c*np.cos(PD_C)*X_C + k_d*np.cos(PD_D)*X_D
    
    g1 = h*(1/tao_c)*((G_this - a)*X_this - inj_term) #h * dE1/dt
    X_thisa = X_this + a1*g1

    g2 = h*(1/tao_c)*((G_this - a)*X_thisa - inj_term)
    X_thisb = X_this + b1*g1 + b2*g2
    
    g3 = h*(1/tao_c)*((G_this - a)*X_thisb - inj_term)
    X_thisc = X_this + c1*g1 + c2*g2 + c3*g3
    
    g4 = h*(1/tao_c)*((G_this - a)*X_thisc - inj_term)
    X_thisd = X_this + d1*g1 + d2*g2 + d3*g3 + d4*g4
    
    g5 = h*(1/tao_c)*((G_this - a)*X_thisd - inj_term)
    X_thise = X_this + e1*g1 + e2*g2 + e3*g3 + e4*g4 + e5*g5
    
    g6 = h*(1/tao_c)*((G_this - a)*X_thise - inj_term)
    
    RK4 = X_this + fo1*g1 + fo3*g3 + fo4*g4 + fo5*g5
    RK5 = X_this + fi1*g1 + fi3*g3 + fi4*g4 + fi5*g5 + fi6*g6
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

def GRK45(X_this, G_this, h, args, coeffs):

    tao_c, tao_f, a, p, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs

    #G1#
    g1 = h*((p/tao_f) - (G_this/tao_f)*(1 + X_this**2)) #h * dG1/dt
    G_thisa = G_this + a1*g1

    g2 = h*((p/tao_f) - (G_thisa/tao_f)* (1 + X_this**2))
    G_thisb = G_this + b1*g1 + b2*g2

    g3 = h*((p/tao_f) - (G_thisb/tao_f)* (1 + X_this**2))
    G_thisc = G_this + c1*g1 + c2*g2 + c3*g3

    g4 = h*((p/tao_f) - (G_thisc/tao_f)* (1 + X_this**2))
    G_thisd = G_this + d1*g1 + d2*g2 + d3*g3 + d4*g4

    g5 = h*((p/tao_f) - (G_thisd/tao_f)* (1 + X_this**2))
    G_thise = G_this + e1*g1 + e2*g2 + e3*g3 + e4*g4 + e5*g5

    g6 = h*((p/tao_f) - (G_thise/tao_f)* (1 + X_this**2))

    RK4 = G_this + fo1*g1 + fo3*g3 + fo4*g4 + fo5*g5
    RK5 = G_this + fi1*g1 + fi3*g3 + fi4*g4 + fi5*g5 + fi6*g6
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
    
def DPRK45(X_this, X_that, DP, k, delta_w, h, args, coeffs):
    tao_c, tao_f, a, p, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs

    #deltaphi#
    factor = (k/tao_c)*(X_this/X_that + X_that/X_this)
    
    g1 = h*(factor*np.sin(DP) + delta_w)
    DPa = DP + a1*g1

    g2 = h*(factor*np.sin(DPa) + delta_w)
    DPb = DP + b1*g1 + b2*g2

    g3 = h*(factor*np.sin(DPb) + delta_w)
    DPc = DP + c1*g1 + c2*g2 + c3*g3

    g4 = h*(factor*np.sin(DPc) + delta_w)
    DPd = DP + d1*g1 + d2*g2 + d3*g3 + d4*g4

    g5 = h*(factor*np.sin(DPd) + delta_w)
    DPe = DP + e1*g1 + e2*g2 + e3*g3 + e4*g4 + e5*g5

    g6 = h*(factor*np.sin(DPe) + delta_w)
    
    RK4 = DP + fo1*g1 + fo3*g3 + fo4*g4 + fo5*g5
    RK5 = DP + fi1*g1 + fi3*g3 + fi4*g4 + fi5*g5 + fi6*g6
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

def timeEvolutionRK45(variablelists0, kdwlists0, h0, timestop, args, stabCond, ssCheck, n):
    """
    Plots evolution of mutually coupled lasers
    using RK45 method with variable timesteps and steady state detection
    see Mutual Injection-Locking and Coherent Combining Chen et. al equation(16)
    
    input:
        Alllists0: Initial E, G, DP, k, dw values
        h0: Initial time step (ns)
        timestop: End of simulation
        args: arguments
        stabCond: stability conditions, end sim after met
        ssCheck: Check if steady state is obtained
        n: size of laser array (n x n)
    """
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, \
        fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 \
        = 1./4, 3./32, 9./32, 1932./2197., -7200./2197, 7296./2197, \
          439./216, -8, 3680./513, -845./4104, \
          -8./27, 2, -3544./2565, 1859./4104, -11./40., \
          25./216, 0., 1408./2565, 2197./4104, -1./5, \
          16./135, 0., 6656./12825, 28561./56430, -90./50, 2./55

    coeffs = a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, \
        fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6

    tao_c, tao_f, a, p, esp, hmin, hmax = args
    
    arguments = tao_c, tao_f, a, p, esp, hmin, hmax

    Xlists = defaultdict(list)
    Glists = defaultdict(list)
    DPrlists = defaultdict(list)
    DPblists = defaultdict(list)
    krValues = defaultdict(list)
    kbValues = defaultdict(list)
    dwrValues = defaultdict(list)
    dwbValues = defaultdict(list)
    timelist = []
    hlist = []
    timelist.append(0)
    hlist.append(0)
    
    variablelists = [Xlists, Glists, DPrlists, DPblists]
    kdwlists = [krValues, kbValues, dwrValues, dwbValues]
    thlists = [timelist, hlist]

#Initialising initial Values
    for i in range(n): #Laser element row number
        for j in range(n): #Laser element column number
            for k in range(4):
                variablelists[k][j+i*10].append(variablelists0[k][j+i*10][0])
                kdwlists[k][j+i*10].append(kdwlists0[k][j+i*10][0])

    for k in range(n):
        for l in range(4):
            variablelists[l][k*10-1].append(0.)
            variablelists[l][k-10].append(0.)
            variablelists[l][k*10+n].append(0.)
            variablelists[l][k+10*n].append(0.)

            kdwlists[l][k*10-1].append(0.)
            kdwlists[l][k-10].append(0.)
            kdwlists[l][k*10+n].append(0.)
            kdwlists[l][k+10*n].append(0.)
    
    hcurrent = h0
    timecurrent = 0.
    X1steadyStateBreak = 0.
    DPsteadyStateBreak = 0.

    while timecurrent < timestop:
        hnewlist = []
        for i in range(n):
            for j in range(n):
                #Get current values with relevance to a single laser from last entry in lists
                xthis, xa, xb, xc, xd, gthis, dpa, dpb, dpc, dpd, ka, kb, kc, kd, dwa, dwb, dwc, dwd =\
                variablelists[0][j+i*10][-1], variablelists[0][j+(i-1)*10][-1], variablelists[0][j+(i+1)*10][-1], variablelists[0][(j-1)+i*10][-1], variablelists[0][(j+1)+i*10][-1],\
                variablelists[1][j+i*10][-1], \
                -(variablelists[3][j+(i-1)*10][-1]), variablelists[3][(j+i*10)][-1], -(variablelists[2][(j-1)+i*10][-1]), variablelists[2][(j+i*10)][-1],\
                (kdwlists[1][j+(i-1)*10][-1]), kdwlists[1][(j+i*10)][-1], (kdwlists[0][(j-1)+i*10][-1]), kdwlists[0][(j+i*10)][-1],\
                -(kdwlists[3][j+(i-1)*10][-1]), kdwlists[3][(j+i*10)][-1], -(kdwlists[2][(j-1)+i*10][-1]), kdwlists[2][(j+i*10)][-1]
                
                #Determine min value of h for a single laser#
                XRK4, XRK5, Xhnew, Xdiff = ERK45(xthis, xa, xb, xc, xd, gthis, dpa, dpb, dpc, dpd, ka, kb, kc, kd, hcurrent, arguments, coeffs)
                GRK4, GRK5, Ghnew, Gdiff = GRK45(xthis, gthis, hcurrent, arguments, coeffs)
                hnewlist.append(Xhnew)
                hnewlist.append(Ghnew)

                if(j<(n-1)):
                    DPrRK4, DPrRK5, DPrhnew, DPrdiff = DPRK45(xthis, xd, dpd, kd, dwd, hcurrent, arguments, coeffs)
                    hnewlist.append(DPrhnew)
                if(i<(n-1)):
                    DPbRK4, DPbRK5, DPbhnew, DPbdiff = DPRK45(xthis, xb, dpb, kb, dwb, hcurrent, arguments, coeffs)
                    hnewlist.append(DPbhnew)
                
        hcurrent = np.nanmin(hnewlist)
        if np.isnan(hcurrent):
            hcurrent = hmax
            thlists[1].append(hmax)
        else:
            thlists[1].append(hcurrent)
        
        #Time#
        timecurrent += hcurrent
        thlists[0].append(timecurrent)
        
        #Calculates values for t+h
        for i in range(n):
            for j in range(n):
                #Get current values with relevance to a single laser from last entry in lists
                xthis, xa, xb, xc, xd, gthis, dpa, dpb, dpc, dpd, ka, kb, kc, kd, dwa, dwb, dwc, dwd =\
                variablelists[0][j+i*10][-1], variablelists[0][j+(i-1)*10][-1], variablelists[0][j+(i+1)*10][-1], variablelists[0][(j-1)+i*10][-1], variablelists[0][(j+1)+i*10][-1],\
                variablelists[1][j+i*10][-1], \
                -(variablelists[3][j+(i-1)*10][-1]), variablelists[3][(j+i*10)][-1], -(variablelists[2][(j-1)+i*10][-1]), variablelists[2][(j+i*10)][-1],\
                (kdwlists[1][j+(i-1)*10][-1]), kdwlists[1][(j+i*10)][-1], (kdwlists[0][(j-1)+i*10][-1]), kdwlists[0][(j+i*10)][-1],\
                -(kdwlists[3][j+(i-1)*10][-1]), kdwlists[3][(j+i*10)][-1], -(kdwlists[2][(j-1)+i*10][-1]), kdwlists[2][(j+i*10)][-1]

                Xnext, XRK5, Xhnew, Xdiff = ERK45(xthis, xa, xb, xc, xd, gthis, dpa, dpb, dpc, dpd, ka, kb, kc, kd, hcurrent, arguments, coeffs)
                if not math.isinf(Xnext):
                    variablelists[0][j+i*10].append(Xnext)
                    xthis = Xnext
                else:
                    return variablelists, kdwlists, thlists

                Gnext, GRK5, Ghnew, Gdiff = GRK45(xthis, gthis, hcurrent, arguments, coeffs)
                if not math.isinf(Gnext):
                    variablelists[1][j+i*10].append(Gnext)
                    gthis = Gnext
                else:
                    return variablelists, kdwlists, thlists

        for i in range(n):
            for j in range(n):
                #Get current values with relevance to a single laser from last entry in lists
                xthis, xa, xb, xc, xd, gthis, dpa, dpb, dpc, dpd, ka, kb, kc, kd, dwa, dwb, dwc, dwd =\
                variablelists[0][j+i*10][-1], variablelists[0][j+(i-1)*10][-1], variablelists[0][j+(i+1)*10][-1], variablelists[0][(j-1)+i*10][-1], variablelists[0][(j+1)+i*10][-1],\
                variablelists[1][j+i*10][-1], \
                -(variablelists[3][j+(i-1)*10][-1]), variablelists[3][(j+i*10)][-1], -(variablelists[2][(j-1)+i*10][-1]), variablelists[2][(j+i*10)][-1],\
                (kdwlists[1][j+(i-1)*10][-1]), kdwlists[1][(j+i*10)][-1], (kdwlists[0][(j-1)+i*10][-1]), kdwlists[0][(j+i*10)][-1],\
                -(kdwlists[3][j+(i-1)*10][-1]), kdwlists[3][(j+i*10)][-1], -(kdwlists[2][(j-1)+i*10][-1]), kdwlists[2][(j+i*10)][-1]
                if(j<(n-1)):
                    DPrnext, DPrRK5, DPrhnew, DPrdiff = DPRK45(xthis, xd, dpd, kd, dwd, hcurrent, arguments, coeffs)
                    if not math.isnan(DPrnext):
                        variablelists[2][j+i*10].append(divmod(DPrnext,(2*np.pi))[1])
                    else:
                        return variablelists, kdwlists, thlists
                else:
                    DPrnext = 0.
                    variablelists[2][j+i*10].append(divmod(DPrnext,(2*np.pi))[1])

                if(i<(n-1)):
                    DPbnext, DPbRK5, DPbhnew, DPbdiff = DPRK45(xthis, xb, dpb, kb, dwb, hcurrent, arguments, coeffs)
                    if not math.isnan(DPbnext):
                        variablelists[3][j+i*10].append(divmod(DPbnext,(2*np.pi))[1])
                    else:
                        return variablelists, kdwlists, thlists
                else:
                    DPbnext = 0.
                    variablelists[3][j+i*10].append(divmod(DPbnext,(2*np.pi))[1])
                        
        #############Check for steady state of centre laser to end sim early############
        index = int(round(n/2.))
        Xcentrelen = len(variablelists[0][(index-1)*11])
        DPcentrelen = len(variablelists[2][(index-1)*11])
        if(Xcentrelen > 1000) and (timecurrent > tao_c*1e4):
            X1_ss_para = np.std(10*np.log10(scipy.array(variablelists[0][index*11][-1000:-1]))) #Check for steady state to end sim prematurely
            if X1_ss_para < ssCheck[0]:
                DP_ss_para = np.std(scipy.array(variablelists[2][index*11])) #Check for steady state to end sim prematurely
##                print X1_ss_para, DP_ss_para
                if DP_ss_para < ssCheck[1]:
                    return variablelists, kdwlists, thlists

    ############Output lists at end of timestop##############
    return variablelists, kdwlists, thlists

def singlelaser(variablelists0, kdwlists0, args, n, plot='yes'):
    variablelists, kdwlists, thlists = timeEvolutionRK45(variablelists0, kdwlists0, h0, timestop, args, stabCond, ssCheck, n)

    alllists = []
    for i in range(n):
        for j in range(n):
            for a in range(4):
                alllists.append(variablelists[a][j+i*10])

    alllists.append(thlists[0])
    alllists.append(thlists[1])

    alllists = samelength(alllists)
    
    for a in range(4):
        for i in range(n):
            for j in range(n):
                for b in range(2):
                    variablelists[a][j+i*10].pop()

    for c in range(2):                    
        for d in range(2):
            thlists[c].pop()

    index = int(round(n/2.))
    stablelist = []
    for i in range(3):
        stablelist.append(np.std(scipy.array(variablelists[i][(index-1)*11][-50:-1])))

    print stablelist[2], stablelist[0], stablelist[1]
    
    if (stablelist[2] < stabCond[2]) and (stablelist[0] < stabCond[0]) and (stablelist[1] < stabCond[1]):
        print 'stable'
    else:
        print 'unstable'

    Xlist = variablelists[0][(index-1)*11]
    Glist = variablelists[1][(index-1)*11]
    timelist = thlists[0]

    Xss = Xlist[-1]
    Gss = Glist[-1]
    
    print 'Single Laser E is %f' %abs(Xss)
    print 'Single Laser G is %f' %abs(Gss)
    
    if plot == 'yes':
        plt.plot(timelist, Xlist,label='Single laser')
        plt.legend()
        plt.show()
        return Xss, Gss
    else:
        return Xss, Gss

def plotEvolution(variablelists0, kdwlists0, args, stabCond, ssCheck, n):
    index = int(round(n/2.))
    kr = kdwlists0[0][(index-1)*11][0]
    dwr = kdwlists0[2][(index-1)*11][0]
    if kr != 0.0:
        print dwr*tao_c/(2*kr)
    print kr, dwr
    
    variablelists, kdwlists, thlists = timeEvolutionRK45(variablelists0, kdwlists0, h0, timestop, args, stabCond, ssCheck, n)
    
    alllists = []
    for i in range(n):
        for j in range(n):
            for a in range(4):
                alllists.append(variablelists[a][j+i*10])

    alllists.append(thlists[0])
    alllists.append(thlists[1])
    alllists = samelength(alllists)
    
    for a in range(4):
        for i in range(n):
            for j in range(n):
                for b in range(2):
                    variablelists[a][j+i*10].pop()

    for c in range(2):                    
        for d in range(2):
            thlists[c].pop()

    index = int(round(n/2.))
    stablelist = []
    for i in range(3):
        stablelist.append(np.std(scipy.array(variablelists[i][(index-1)*11][-50:-1])))

    print stablelist[2], stablelist[0], stablelist[1]
    
    if (stablelist[2] < stabCond[2]) and (stablelist[0] < stabCond[0]) and (stablelist[1] < stabCond[1]):
        print 'stable'
    else:
        print 'unstable'

    timelist = thlists[0]

    #Normalise X magnitude to that of single laser and convert to dB
    label =  ' $\kappa$ = %1.3f, $\Delta\omega$ = %1.0f' %(kr, dwr*1e3)

    plt.figure(1)
    for i in range(n):
        for j in range(n):
            Xlist = variablelists[0][j+i*10]
            Xnorm = (10* np.log10(scipy.array(Xlist)/Esl))    
            traceLabel = 'E %0.0f' % (j+i*10)
            plt.plot(timelist, Xnorm,label=traceLabel)
            plt.legend(loc=4,prop={'size': 10})
            plt.xlabel('Time (ps)')
            plt.ylabel('$E/E_{single laser}$ (dB)')
            plt.title('Time evolution of E field amplitude')

        
    plt.figure(2)
    for i in range(n):
        for j in range(n):
            Glist = variablelists[1][j+i*10]
            traceLabel = 'G %0.0f' % (j+i*10)
            plt.plot(timelist, scipy.array(Glist),label=traceLabel)
            plt.legend(prop={'size': 10})
            plt.xlabel('Time (ps)')
            plt.ylabel('Gain')
            plt.title('Time evolution of laser gain')
    
    plt.figure(3)
    for i in range(n):
        for j in range(n):
            PDrlist = variablelists[2][j+i*10]
            traceLabel = 'PDr %0.0f' % (j+i*10)
            plt.plot(timelist, (scipy.array(PDrlist)/np.pi), label=traceLabel)
            plt.legend(loc=4,prop={'size': 10})
            plt.xlabel('Time (ps)')
            plt.ylabel('Phase difference ($\Delta\phi/\pi$)')
            plt.title('Time evolution of phase difference')

    plt.figure(4)
    for i in range(n):
        for j in range(n):
            PDblist = variablelists[3][j+i*10]
            traceLabel = 'PDb %0.0f' % (j+i*10)
            plt.plot(timelist, (scipy.array(PDblist)/np.pi), label=traceLabel)
            plt.legend(loc=4,prop={'size': 10})
            plt.xlabel('Time (ps)')
            plt.ylabel('Phase difference ($\Delta\phi/\pi$)')
            plt.title('Time evolution of phase difference')


    plt.figure(5)
    hlist = thlists[1]
    plt.plot(timelist,scipy.array(hlist))
    plt.xlabel('Time (ps)')
    plt.ylabel('Time step (ps)')
    plt.title('Time evolution of timestep')

    plt.figure(6)
    for i in range(n):
        for j in range(n):
            Xlist = variablelists[0][j+i*10]
            traceLabel = 'E %0.0f' % (j+i*10)
            plt.plot(timelist, Xlist, label=traceLabel)
            plt.legend(prop={'size': 10})
            plt.xlabel('Time (ps)')
            plt.ylabel('$E/E_{single laser}$')
            plt.title('Time evolution of E field amplitude')

    plt.figure(7)
    Xlist = variablelists[0][(index-1)*11]
    Xnorm = (10* np.log10(scipy.array(Xlist)/Esl))    
    traceLabel = 'E %0.0f' % (1)
    plt.plot(timelist, Xlist,label=traceLabel)
    plt.legend(loc=4,prop={'size': 10})
    plt.xlabel('Time (ps)')
    plt.ylabel('$E/E_{single laser}$ (dB)')
    plt.title('Time evolution of E field amplitude' + label)

    plt.figure(8)
    PDrlist = variablelists[2][(index-1)*11]
    traceLabel = 'PDr %0.0f' % (j+i*10)
    plt.plot(timelist, (scipy.array(PDrlist)/np.pi), label=traceLabel)
    plt.legend(loc=4,prop={'size': 10})
    plt.xlabel('Time (ps)')
    plt.ylabel('Phase difference ($\Delta\phi/\pi$)')
    plt.title('Time evolution of phase difference')

    return variablelists, thlists

#######################Main###########################
L = 5*1e6 #Optical length of one round trip (pm)
c = 3e8 #Speed of light (pm/ps)
tao_c = 2*L/c #Round trip time (ps)
tao_f = 1e3 #Fluorescence time (ps)
a = 0.1 #cavity loss coefficient (loss per oscillation)
p = 2. #pumping coefficient

#############################RK4 parameters############################
n = 4 #Size of laser array
timestop = 500 #(ps)
h0 = 1e-3 #(ps) Starting timestep
esp = [0.05, 1e-3, 0.005] #Error allowance parameter for [X, G, DP]
hmin = 1e-4 #Min timestep
hmax = 1e-2 #Max timestep
stabCond = [0.3, 1e-3, 0.03] #Stability conditions for [X,G,DP]
ssCheck = [0.2, 1e-4] #Steady state detector paramters for [X, DP]
injRatioRange = [-25, -23] #Injection ratio range(dB)
injRatioNPoints = 30
dwSpacing = 0.05
args = tao_c, tao_f, a, p, esp, hmin, hmax 

k_range = [0.0034, 0.005]
dw_range = [0., 0.2]

############################Alllists initialisation#########################
X0 = defaultdict(list)
G0 = defaultdict(list)
DPr0 = defaultdict(list)
DPb0 = defaultdict(list)
kr0 = defaultdict(list)
kb0 = defaultdict(list)
dwr0 = defaultdict(list)
dwb0 = defaultdict(list)
variablelists0 = [X0, G0, DPr0, DPb0] 
kdwlists0 = [kr0, kb0, dwr0, dwb0]

for i in range(n): #Laser element row number
    for j in range(n): #Laser element column number
        variablelists0[0][j+i*10].append(1.) #X0 values
        variablelists0[1][j+i*10].append(2.) #G0 values
        
        if (j<(n-1)):
            variablelists0[2][j+i*10].append(random.uniform(0., 2*np.pi)) #DPr values
            kdwlists0[0][j+i*10].append(random.uniform(k_range[0], k_range[1])) #kr values
            kdwlists0[2][j+i*10].append(random.uniform(dw_range[0], dw_range[1])) #dwr values
        else:
            variablelists0[2][j+i*10].append(0.) #DPr values
            kdwlists0[0][j+i*10].append(0.) #kr values
            kdwlists0[2][j+i*10].append(0.) #dwr values

        if (i<(n-1)):
            variablelists0[3][j+i*10].append(random.uniform(0., 2*np.pi)) #DPb values
            kdwlists0[1][j+i*10].append(random.uniform(k_range[0], k_range[1])) #kb values
            kdwlists0[3][j+i*10].append(random.uniform(dw_range[0], dw_range[1])) #dwb values
        else:
            variablelists0[3][j+i*10].append(0.) #DPb values
            kdwlists0[1][j+i*10].append(0.) #kr values
            kdwlists0[3][j+i*10].append(0.) #dwr values

############################Running Simulations#########################
##print 'Calculating single laser'
##E,G = singlelaser(variablelists0, kdwlists0, args, n)
##Esl,Gsl = abs(E), abs(G)
##print Esl, Gsl

Esl, Gsl = 4.3, 0.1 

print 'Calculating Time evolution'
variablelists, thlists = plotEvolution(variablelists0, kdwlists0, args, stabCond, ssCheck, n) #(k,delta_w(THz))    
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
