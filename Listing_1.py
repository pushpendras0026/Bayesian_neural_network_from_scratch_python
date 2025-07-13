# random number generation from a gaussian distribution
import numpy as np
from numpy import  random
from scipy import stats
#  we are going to draw 10 sample from a normal distrib of N(0,1)
samples=random.randn(10)
# making gaussian distribution pdf with mean mu and std sigma 
x=np.linspace(0,8,1000)
pdf=stats.norm.pdf(x,loc=4,scale=0.5)

