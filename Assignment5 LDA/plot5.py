import matplotlib.pyplot as plt
import numpy as np
non = [141284622.069, 232461039.047, 338761644.158, 299955297.183, 335329925.068]
imposter = [578291138.807, 546883630.225, 510591213.234, 611706203.892]

non.sort()
imposter.sort()
t1 = np.arange(len(non))
t2 = np.arange(len(imposter))
plt.plot(t1, non, color = 'black', label = 'non imposter')
plt.plot(t2, imposter, color = 'red', label = 'imposter')
plt.xlabel('Images')
plt.ylabel('Distance')
plt.title('PCA')
plt.legend()
plt.show()
print()