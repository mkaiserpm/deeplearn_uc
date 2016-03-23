'''
Created on 23.03.2016

@author: mario
'''
# Plot softmax curves
import matplotlib.pyplot as plt
import numpy as np

from deeputils.base import softmax

scores = np.array([3.0, 1.0, 0.2])

print(softmax(scores))
print(softmax(scores*10))
print(softmax(scores/10))
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()