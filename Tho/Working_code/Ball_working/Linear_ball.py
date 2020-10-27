from scipy import stats
import numpy as np
my_y = np.arange(20, 60.1, 1)
my_x = np.array([0.003711952487008, 0.003824091778203, 0.003980891719745, 0.004132231404959, 0.004253509145045,
                 0.004587155963303, 0.004675081813932, 0.004970178926441, 0.00470145745181, 0.005241090146751,
                 0.005086469989827, 0.005344735435596, 0.00551876379691, 0.005656108597285, 0.005727376861397,
                 0.005941770647653, 0.006060606060606, 0.006161429451633, 0.006333122229259, 0.00640204865557,
                 0.00668449197861, 0.006910850034554, 0.006854009595613, 0.006910850034554, 0.007153075822604,
                 0.007147962830593, 0.007293946024799, 0.007473841554559, 0.007501875468867, 0.007716049382716,
                 0.00776397515528, 0.008, 0.00815660685155, 0.008361204013378, 0.008354218880535, 0.00886524822695,
                 0.008554319931565, 0.008802816901408, 0.009025270758123, 0.009041591320072, 0.009496676163343])
slope, intercept, r_value, p_value, std_err = stats.linregress(my_x, my_y)
prediction = my_x * slope + intercept
mean_abs_err = sum(abs(my_y - prediction))/len(my_y)
std_deviation_err = np.sqrt(sum(abs(my_y - prediction))/len(my_y))
print("Slope is: {}".format(slope))
print("Intercept is: {}".format(intercept))
print("Mean absolute error is: {}".format(mean_abs_err))
print("Standard deviation error is: {}".format(std_deviation_err))
print("Maximum error on set: {}".format(abs(my_y - prediction).max()))

