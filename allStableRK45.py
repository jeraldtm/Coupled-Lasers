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
    G1list = lists[2]
    G2list = lists[3]
    deltaphilist = lists[4]
    time = lists[5]

    return X1list, X2list, G1list, G2list, deltaphilist, time

def ERK45(Xthis, Xthat, Gthis, Gthat, deltaphi, h, args, coeffs):

    tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs
    
    #X1#
    k1 = h*(1/tao_c)*((Gthis - a)*Xthis - k*np.cos(deltaphi)*Xthat) #h * dE1/dt
    Xthisa = Xthis + a1*k1
    k2 = h*(1/tao_c)*((Gthis - a)*Xthisa - k*np.cos(deltaphi)*Xthat)
    Xthisb = Xthis + b1*k1 + b2*k2
    k3 = h*(1/tao_c)*((Gthis - a)*Xthisb - k*np.cos(deltaphi)*Xthat)
    Xthisc = Xthis + c1*k1 + c2*k2 + c3*k3
    k4 = h*(1/tao_c)*((Gthis - a)*Xthisc - k*np.cos(deltaphi)*Xthat)
    Xthisd = Xthis + d1*k1 + d2*k2 + d3*k3 + d4*k4
    k5 = h*(1/tao_c)*((Gthis - a)*Xthisd - k*np.cos(deltaphi)*Xthat)
    Xthise = Xthis + e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5
    k6 = h*(1/tao_c)*((Gthis - a)*Xthisd - k*np.cos(deltaphi)*Xthat)
    RK4 = Xthis + fo1*k1 + fo3*k3 + fo4*k4 + fo5*k5
    RK5 = Xthis + fi1*k1 + fi3*k3 + fi4*k4 + fi5*k5 + fi6*k6
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

def GRK45(Xthis, Xthat, Gthis, Gthat, deltaphi, h, args, coeffs):

    tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs

    #G1#
    k1 = h*((p/tao_f) - (Gthis/tao_f)*(1 + Xthis**2)) #h * dG1/dt
    Gthisa = Gthis + a1*k1
    k2 = h*((p/tao_f) - (Gthisa/tao_f)* (1 + Xthis**2))
    Gthisb = Gthis + b1*k1 + b2*k2
    k3 = h*((p/tao_f) - (Gthisb/tao_f)* (1 + Xthis**2))
    Gthisc = Gthis + c1*k1 + c2*k2 + c3*k3
    k4 = h*((p/tao_f) - (Gthisc/tao_f)* (1 + Xthis**2))
    Gthisd = Gthis + d1*k1 + d2*k2 + d3*k3 + d4*k4
    k5 = h*((p/tao_f) - (Gthisd/tao_f)* (1 + Xthis**2))
    Gthise = Gthis + e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5
    k6 = h*((p/tao_f) - (Gthise/tao_f)* (1 + Xthis**2))
    RK4 = Gthis + fo1*k1 + fo3*k3 + fo4*k4 + fo5*k5
    RK5 = Gthis + fi1*k1 + fi3*k3 + fi4*k4 + fi5*k5 + fi6*k6
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
    
def DPRK45(X1, X2, G1, G2, deltaphi, h, args, coeffs):
    tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax = args
    a1, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, fo1, fo2, fo3, fo4, fo5, fi1, fi2, fi3, fi4, fi5, fi6 = coeffs

    #deltaphi#
    k1 = h*((k/tao_c)*(X1/X2 + X2/X1)*np.sin(deltaphi) + delta_w)
    dpa = deltaphi + a1*k1
    k2 = h*((k/tao_c)*(X1/X2 + X2/X1)*np.sin(dpa) + delta_w)
    dpb = deltaphi + b1*k1 + b2*k2
    k3 = h*((k/tao_c)*(X1/X2 + X2/X1)*np.sin(dpb) + delta_w)
    dpc = deltaphi + c1*k1 + c2*k2 + c3*k3
    k4 = h*((k/tao_c)*(X1/X2 + X2/X1)*np.sin(dpc) + delta_w)
    dpd = deltaphi + d1*k1 + d2*k2 + d3*k3 + d4*k4
    k5 = h*((k/tao_c)*(X1/X2 + X2/X1)*np.sin(dpd) + delta_w)
    dpe = deltaphi + e1*k1 + e2*k2 + e3*k3 + e4*k4 + e5*k5
    k6 = h*((k/tao_c)*(X1/X2 + X2/X1)*np.sin(dpe) + delta_w)
    RK4 = deltaphi + fo1*k1 + fo3*k3 + fo4*k4 + fo5*k5
    RK5 = deltaphi + fi1*k1 + fi3*k3 + fi4*k4 + fi5*k5 + fi6*k6
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

def timeEvolutionRK45(X0, G0, phi0, h0, timestop, args):
    """
    Plots evolution of mutually coupled lasers
    using RK4 method
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
    G1list = []
    G2list = []
    deltaphilist = []
    timelist = []
    
    X1list.append(X0[0][0])
    X2list.append(X0[1][0])
    G1list.append(G0[0][0])
    G2list.append(G0[1][0])    
    deltaphilist.append(phi0)
    timelist.append(0)
    
    tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax = args       
    timecurrent = 0
                         
    while timecurrent < timestop:
        x1, x2, g1, g2, dp = X1list[-1], X2list[-1], G1list[-1], G2list[-1], deltaphilist[-1]
        #Determine min value of h#
        X1RK4, X1RK5, X1hnew, X1diff = ERK45(x1, x2, g1, g2, dp, hcurrent, args, coeffs)
        G1RK4, G1RK5, G1hnew, G1diff = GRK45(x1, x2, g1, g2, dp, hcurrent, args, coeffs)
        X2RK4, X2RK5, X2hnew, X2diff = ERK45(x2, x1, g2, g1, dp, hcurrent, args, coeffs)
        G2RK4, G2RK5, G2hnew, G2diff = GRK45(x2, x1, g2, g1, dp, hcurrent, args, coeffs)
        DPRK4, DPRK5, DPhnew, DPdiff = DPRK45(x1, x2, g1, g2, dp, hcurrent, args, coeffs)
        hnewlist = [X1hnew, G1hnew, X2hnew, G2hnew, DPhnew]
        difflist = [X1diff, G1diff, X2diff, G2diff, DPdiff]
        hcurrent = np.nanmin(hnewlist)
        if np.isnan(hcurrent):
            hcurrent = hmax
##        print hcurrent, hnewlist
##        print difflist
        
        #Time#
        timecurrent += hcurrent
        timelist.append(timecurrent)
        
        #Calculates values for t+h
        X1next, X1RK5, X1hnew, X1diff = ERK45(x1, x2, g1, g2, dp, hcurrent, args, coeffs)
        if not math.isinf(X1next):
            X1list.append(X1next)
            x1 = X1next
        else:
            lists = [X1list, X2list, G1list, G2list, deltaphilist, timelist]
            return samelength(lists)

        G1next, G1RK5, G1hnew, G1diff = GRK45(x1, x2, g1, g2, dp, hcurrent, args, coeffs)
        if not math.isinf(G1next):
            G1list.append(G1next)
            g1 = G1next
        else:
            lists = [X1list, X2list, G1list, G2list, deltaphilist, timelist]
            return samelength(lists)
        
        X2next, X2RK5, X2hnew, X2diff = ERK45(x2, x1, g2, g1, dp, hcurrent, args, coeffs)
        if not math.isinf(X2next):
            X2list.append(X2next)
            x2 = X2next
        else:
            lists = [X1list, X2list, G1list, G2list, deltaphilist, timelist]
            return samelength(lists)

        G2next, G2RK5, G2hnew, G2diff = GRK45(x2, x1, g2, g1, dp, hcurrent, args, coeffs)
        if not math.isinf(G2next):
            G2list.append(G2next)
            g2 = G2next
        else:
            lists = [X1list, X2list, G1list, G2list, deltaphilist, timelist]
            return samelength(lists)

        DPnext, DPRK5, DPhnew, DPdiff = DPRK45(x1, x2, g1, g2, dp, hcurrent, args, coeffs)
        if not math.isnan(DPnext):
            deltaphilist.append(divmod(DPnext,(2*np.pi))[1])
            dp = DPnext
        else:
            lists = [X1list, X2list, G1list, G2list, deltaphilist, timelist]
            return samelength(lists)

    lists = [X1list, X2list, G1list, G2list, deltaphilist, timelist]
    return samelength(lists)

def singlelaser():
    args = tao_c, tao_f, a, p, 0.0, 0.0, esp, hmin, hmax 
    
    X1list, X2list, G1list, G2list, deltaphilist, timelist = EfieldRK4(E0, G0, phi0, h, timestop, args)

    for i in range(2):
        X1list.pop()
        X2list.pop()
        G1list.pop()
        G2list.pop()
        deltaphilist.pop()
        timelist.pop()

    stable = abs(deltaphilist[-1] - deltaphilist[-10])
    stable1 = abs(X1list[-1] - X1list[-10])
    stable2 = abs(G1list[-1] - G1list[-10])

    print stable, stable1, stable2
    
    if (stable < 0.0001):
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

def evolution(k,delta_w, stabCond, plotLabel):
    args = tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax 
    print k, delta_w
    if k != 0.0:
        print delta_w*tao_c/(2*k)
    
    X1list, X2list, G1list, G2list, deltaphilist, timelist = timeEvolutionRK45(E0, G0, phi0, h0, timestop, args)

    for i in range(2):
        X1list.pop()
        X2list.pop()
        G1list.pop()
        G2list.pop()
        deltaphilist.pop()
        timelist.pop()

    stable = np.std(scipy.array(deltaphilist[-50:-1]))
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
    X2norm = 10* np.log10(scipy.array(X1list)/Esl)

##    label =  ' $\kappa$ = %1.1f, $\Delta\omega$ = %1.0f GHz' %(k, delta_w*1e3)
    label =  ' $\Delta\omega$ = 2000 GHz'
    
##    plt.figure(1)
##    plt.plot(timelist, X1norm,label='X1')
##    plt.plot(timelist, X2norm,label='X2')
##    plt.legend()
##    plt.xlabel('Time (ps)')
##    plt.ylabel('$E/E_{single laser}$ (dB)')
##    plt.title('Time evolution of E field amplitude' + label)
##        
##    plt.figure(2)
##    plt.plot(timelist, G1list,label='G1')
##    plt.plot(timelist, G2list,label='G2')
##    plt.xlabel('Time (ps)')
##    plt.ylabel('Gain')
##    plt.title('Time evolution of laser gain' + label)
    
    plt.figure(3)
    plt.plot(timelist,scipy.array(deltaphilist)/(np.pi), label = plotLabel)
    plt.xlabel('Time (ps)')
    plt.ylabel('Phase difference ($\Delta\phi/\pi$)')
    plt.title('Time evolution of phase difference' + label)

##    Ess = X1list[-1]
##    Eratio = Ess/Esl 
##    print 'Steady state E field is %f x E_sl' %(Eratio)
    
    return X1list, X2list, G1list, G2list, deltaphilist, timelist 
    
def plot(path, knpoints, dwspacing, stabCond):
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

    for j in np.linspace(-23,-3,knpoints):
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
        print x, kdb, k

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
            args = tao_c, tao_f, a, p, k, delta_w, esp, hmin, hmax 
            X1list, X2list, G1list, G2list, deltaphilist, timelist = \
                    timeEvolutionRK45(E0, G0, phi0, h0, timestop, args)

            #Remove last element in lists to eliminate edge effects#
            for i in range(2):
                X1list.pop()
                X2list.pop()
                G1list.pop()
                G2list.pop()
                deltaphilist.pop()
                timelist.pop()
            
            stable = np.std(scipy.array(deltaphilist[-50:-1]))
            stable1 = np.std(scipy.array(X1list[-50:-1]))
            stable2 = np.std(scipy.array(G1list[-50:-1]))

            print stable, stable1, stable2
            
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

#############################RK4 parameters############################
timestop = 100 #(ps)
h0 = 1e-2 #(ps) Starting timestep
E0 = [[1.],[1.]]
G0 = [[2.],[2.]]
phi0 = 0.0
esp = [0.01, 1e-4, 0.001] #Accuracy parameter for [X, G, DP]
hmin = 1e-3 #Min timestep
hmax = 1e-2 #Max timestep
stabCond = [0.05, 1e-7, 0.01] #Stability conditions for [X,G,DP]

############################Running Simulations#########################

##print 'Calculating single laser'
##E,G = singlelaser()
##Esl,Gsl = abs(E), abs(G)
##print

Esl, Gsl = 4.3, 0.1 

print 'Calculating Mutual'

kdbstable = -14.
kdbunstable = -16.

kstable = 10**(kdbstable/10.)
kunstable = 10**(kdbunstable/10.)

stableLabel = "k = %1.0f dB" % kdbstable
unstableLabel = "k = %1.0f dB" % kdbunstable

X1list, X2list, G1list, G2list, deltaphilist, timelist = evolution(kstable, 2, stabCond, stableLabel) #(k,delta_w(THz))
X1list, X2list, G1list, G2list, deltaphilist, timelist = evolution(kunstable, 2, stabCond, unstableLabel) #(k,delta_w(THz))    
print

##for i in np.linspace(11.58048122,11.58048133,10):
##    evolution(0.2, i)
##    print

##plot('phasestable11.xls', 100, 0.05)
plt.legend()
plt.show()

#Plot stable region#
##kdb = np.linspace(-20,-6,100)
##deltaw = 2*10**(scipy.array(kdb)/10)*1e3/tao_c
##plt.plot(kdb,deltaw)
##plt.plot(kdb, -deltaw)
##plt.xlabel('Injection Ratio (dB)')
##plt.ylabel('Frequency Detuning (GHz)')
##plt.title('Mutual Injection stable region')
##plt.show()
