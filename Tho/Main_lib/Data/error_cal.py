import numpy as np
error = [0.050000000000001, -4.62, -0.91, -0.760000000000002, 1.19, -2.82, 0.940000000000001, 4.81,
         0.949999999999999, 4.25, 1.2, -0.629999999999999, 2.09, 0.25, 5.14, -0.090000000000003,
         0.25]
standard = np.std(error, dtype=np.float64)
mean = np.mean(error)
print(standard)
print(mean)
