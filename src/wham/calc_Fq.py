#!/usr/bin/python
import numpy as np #to access np.exp() not built int exp
from math import *
#import timeseries # for timeseries analysis 
#import commands
#import pdb;
import pickle
import os.path
import histogram_reweighting2d as WHAM
import load_data
import getopt, sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.pyplot import *
#mbar = pickle.load(open("mbar.pickle","rb"))

#parser = argparse.ArgumentParser(description='Combine energy and overlap data from multiple runs at different temperatures into one histogram and print F_q.')
#parser.add_argument('-f', type=int, nargs='1', help='the number of free particles')
#parser.parse_args()

def usage():
    print sys.argv[0], " [-hF -o output_prefix -r rskip -q qcolumn -e ecolumn -E nebins -c input -T TRANGE ] -f nfree"
    print 'Combine energy and overlap data from multiple runs at different temperatures into one histogram and print various quantities.'
    print '  -h print this help and exit'
    print '  -f nfree : number of mobile particles, used to determine nqbins and # degrees of freedom'
    print '  -o output_prefix : change the default output_prefix'
    print '  -F :  dont use pickle file'
    print '  -r rskip : skip the first fraction r of the data files'
    print '  -q qcolumn : which column to get the overlap data from'
    print '  -e ecolumn : which column to get the energy data from'
    print '  -E nebins  : number of energy bins (=300)'
    print '  -c input : Make a linear combination of two order parameters.'
    print '             Input will have the form "q1column q2column q2weight"'
    print '             The order parameter will be q = (q1 + q2weight*q2)/(1+q2weight)'
    print '  -T TRANGE : set TRANGE for the calculation of Fq.  TRANGE should have the format "Tmin Tmax numT"'

usepkl=True
nfree=0;
outprefix="out"
rskip=0.0
qcolumn=3
ecolumn=2
qcombine=[]
TRANGEi=[]
nebins=300
dEmin=0.2


opts, args = getopt.getopt(sys.argv[1:], "hf:o:Fr:q:c:T:e:E:", ["help", "nfree="])
output = None
verbose = False
for o, a in opts:
    if o == "-f":
        nfree=int(a)
    elif o == "-o":
        outprefix=a
        print "output_prefix = ", outprefix
    elif o == "-F":
        usepkl=False
    elif o == "-r":
        rskip=float(a)
        print "will skip the first ", rskip, "of the data files"
    elif o == "-q":
        qcolumn=int(a)
        print "using qcolumn = ", qcolumn
    elif o == "-e":
        ecolumn=int(a)
        print "using ecolumn = ", ecolumn
    elif o == "-E":
        nebins=int(a)
        print "using nebins = ", nebins
    elif o == "-c":
        qcombline=a
        qcombine = [float(b) for b in qcombline.split()]
        if len(qcombine) != 3:
            print "-c: qcombine must have 3 parts: ", qcombine
            usage()
            exit(1)
        print "using qcombine: ", qcombine
    elif o == "-T":
        line = [float(b) for b in a.split()]
        if len(line) != 3:
            print "-T: TRANGE must have 3 parts: ", qcombine
            usage()
            exit(1)
        TMIN = line[0]
        TMAX = line[1]
        NTEMP = int(line[2])
        TINT=(TMAX-TMIN)/(NTEMP-1)
        TRANGEi = [ TMIN + i*TINT for i in range(NTEMP) ]
        print "using TRANGE: ", TRANGEi
    elif o in ("-h", "--help"):
        usage()
        sys.exit()
    else:
        assert False, "unhandled option"


if nfree == 0:
    print "nfree must be given"
    usage()
    exit(1)


pklname=outprefix+".pickle"
if not usepkl or not os.path.isfile(pklname):
  
    #get temperatures and filenames
    Tlist=list(np.genfromtxt('temperatures'))
    rep1=0 #don't use the first 8 replicas.  There is a better way to do this.
    Tlist=Tlist[(rep1):]
    nrep = len(Tlist)
    filenames=['overlap.'+str(n+rep1+1) for n in range(nrep)]
  
    #OK, now we have a list of temperatures and filenames for each replicas
    print "replica list:"
    for n in range(nrep):
        print Tlist[n], filenames[n]
  
  
    print "USING nfree = ", nfree
    nqbins=nfree+1
  
  
    data = load_data.loadData2dExp(filenames, ecolumn, qcolumn, nqbins, fskip=rskip, qcombine=qcombine, nebins=nebins, dEmin=dEmin)
  
    wham = WHAM.wham2d(Tlist, data.binenergy, data.binq, data.visits2d)
  
    wham.minimize()
    
     
    print "dumping WHAM2d to pickle file: ", pklname
    pickle.dump(wham,open(pklname,"wb"))

else:
    print "=================================================================="
    print "loading WHAM2d from pickle file: ", pklname
    print "=================================================================="
    wham = pickle.load(open(pklname,"rb")) 

##############################################################################
#put data in form appropriate for scatter plots
##############################################################################
q=[]
e=[]
z=[]
log10visits2d = []
log10n_Eq = []
for j in range((wham.nqbins)):
    for i in range((wham.nebins)):
        if wham.visits2d[i,j,:].sum() > 0: #sum over the replicas
            e.append( wham.binenergy[i] )
            q.append( wham.binq[j] )
            log10visits2d.append( np.log10( wham.visits2d[i,j,:].sum() ) )
            log10n_Eq.append( wham.logn_Eq[i,j] / log(10.)  )

print len(log10visits2d)
##############################################################################
#print data
##############################################################################

fname=outprefix+".visits.q"
print 'writing visits to overlap bins to', fname
with open(fname,"w") as fout:
    np.savetxt(fout,wham.visits2d.sum(0)) #sum over the energy
#now plot it
fname+='.pdf'
plt.xlabel("overlap")
plt.ylabel("log10(visits)")
plt.plot(wham.binq, np.log10(wham.visits2d.sum(0)), '.-')
plt.savefig(fname)
plt.clf()

fname=outprefix+".visits2d"
print 'writing log visits2d to', fname
with open(fname,"w") as fout:
    for i in range(len(q)):
        fout.write( str(q[i])+" "+ str(e[i])+" "+ str(log10visits2d[i]) + "\n" ) 
#now plot it
fname+='.pdf'
plt.xlabel("overlap")
plt.ylabel("markov energy")
plt.scatter(q,e,c=log10visits2d)
cbar = plt.colorbar()
cbar.set_label("log10(visits)")
plt.savefig(fname)
plt.clf()


fname=outprefix+".visits.E"
print 'writing visits to energy bins to', fname
with open(fname,"w") as fout:
    visitsE = wham.visits2d.sum(1)
    for i in range(wham.nebins):
        if visitsE[i,:].sum() > 0:
            val = visitsE[i,:].sum()
            vallist = (visitsE[i,:].tolist())
            slist = [ str(v) for v in vallist]
            fout.write( str(wham.binenergy[i]) +" "+ str(val) +" ".join(slist) + "\n" ) 
#now plot it
fname+='.pdf'
plt.xlabel("energy")
plt.ylabel("log10(visits)")
plt.plot(wham.binenergy, np.log10(wham.visits2d.sum(1)), '.-') #sum over q
plt.savefig(fname)
plt.clf()


fname=outprefix+".n_Eq"
print 'dumping histogram log10( n(E,q) ) to', fname
#fout=open(fname,"w")
#np.savetxt(fout,(wham.logn_Eq))
#fout.close()
with open(fname,"w") as fout:
    for i in range(len(q)):
        fout.write( str(q[i])+" "+ str(e[i])+" "+ str(log10n_Eq[i]) + "\n" ) 
#now plot it
fname+='.pdf'
plt.xlabel("overlap")
plt.ylabel("markov energy")
plt.scatter(q,e,c=log10n_Eq)
cbar=plt.colorbar()
cbar.set_label("log10( n(E,q) )")
plt.savefig(fname)
plt.clf()



fname=outprefix+".Fq"
print 'dumping F(q) to', fname
#TRANGEi=[.8+i*.2 for i in range(5)]
#TRANGEi=[]
TRANGE, F_q = wham.calc_Fq(TRANGEi)
fout=open(fname,"w")
fout.write("#T= "+str(TRANGE)+"\n")
np.savetxt(fout,np.column_stack((wham.binq,F_q)))
fout.close()
#now plot it
fname+=".pdf"
pp=PdfPages(fname)
plt.xlabel("overlap")
plt.ylabel("F(overlap)")
plt.plot(F_q)
plt.legend( [str(T) for T in TRANGE])
#plt.savefig(fname)
pp.savefig()
plt.clf()
plt.xlabel("overlap")
plt.ylabel("F(overlap)")
plt.plot(F_q)
plt.legend( [str(T) for T in TRANGE])
plt.ylim((0,30))
pp.savefig()
pp.close()
plt.clf()


print "calculating average overlap"
TRANGE, qavg = wham.calc_qavg()
fname=outprefix+".qavg"
fout=open(fname,"w")
np.savetxt(fout,np.column_stack((TRANGE,qavg)))
fout.close()
fname+=".pdf"
pp=PdfPages(fname)
plt.xlabel("T")
plt.ylabel("<overlap>")
plt.plot(TRANGE, qavg)
plt.savefig(fname)
plt.clf()

fname=outprefix+".Cv"
print 'writing Cv to', fname
fout=open(fname,"w")
wham.calc_Cv(nfree*3, fout)
fout.close()
