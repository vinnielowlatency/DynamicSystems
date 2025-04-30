import numpy as py
import matplotlib.pyplot as plt

p0 = 5
t = [0]
p = [p0]
r = 0.5

for i in range(1,100):
    t.append(i)
    p.append(p[-1]*(1+r))

plt.plot(p,t)

