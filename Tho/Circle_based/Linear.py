from scipy import stats
import numpy as np
my_y = np.array([20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 39, 40])
my_x = np.array([1/42.2000007629395, 1/43.0999984741211, 1/36.9000015258789, 1/38.7999992370605, 1/34.2999992370605,
        1/35.2000007629395, 1/32.7000007629395, 1/30.7000007629395, 1/30.3999996185303, 1/29,
        1/27.2000007629395, 1/27.5, 1/26.3999996185303, 1/24.1000003814697, 1/24.5, 1/22.2000007629395, 1/22])
slope, intercept, r_value, p_value, std_err = stats.linregress(my_x, my_y)
prediction = my_x * slope + intercept
std_err = sum(abs(my_y - prediction))/len(my_y)
print(std_err)
