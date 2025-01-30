#!/usr/bin/env python
# coding: utf-8

sequence_lengths = list(range(5, 31))
accuracy_scores  = [0.850609756097561,0.8445121951219512,0.8414634146341463,0.8323170731707317,0.8536585365853658,
                   0.850609756097561,0.8475609756097561,0.8475609756097561,0.8475609756097561,0.8475609756097561,
                   0.850609756097561,0.8292682926829268,0.8445121951219512,0.850609756097561,0.8353658536585366,
                   0.8384146341463414,0.8384146341463414,0.8445121951219512,0.8353658536585366,0.8445121951219512,
                   0.8445121951219512,0.8445121951219512,0.8323170731707317,0.8414634146341463,0.8384146341463414, 
                   0.8414634146341463]

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

sequence_lengths = list(range(5, 31))
accuracy_scores = [0.850609756097561, 0.8445121951219512, 0.8414634146341463, 0.8323170731707317, 0.8536585365853658,
                   0.850609756097561, 0.8475609756097561, 0.8475609756097561, 0.8475609756097561, 0.8475609756097561,
                   0.850609756097561, 0.8292682926829268, 0.8445121951219512, 0.850609756097561, 0.8353658536585366,
                   0.8384146341463414, 0.8384146341463414, 0.8445121951219512, 0.8353658536585366, 0.8445121951219512,
                   0.8445121951219512, 0.8445121951219512, 0.8323170731707317, 0.8414634146341463, 0.8384146341463414, 
                   0.8414634146341463]

plt.plot(sequence_lengths, accuracy_scores, marker='o')
plt.title('Accuracy vs Sequence Length (NAS)')
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("../outputs/images/acc-vs-seq-len.png", format="png", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

sequence_lengths = list(range(45, 155, 5))
accuracy_scores = [0.7403846153846154, 0.7740384615384616, 0.75,               0.7788461538461539, 0.7788461538461539, 
                   0.7836538461538461, 0.7836538461538461, 0.7836538461538461, 0.7692307692307693, 0.8076923076923077, 
                   0.8076923076923077, 0.8076923076923077, 0.7644230769230769, 0.8028846153846154, 0.7980769230769231, 
                   0.7740384615384616, 0.7692307692307693, 0.7932692307692307, 0.7884615384615384, 0.7692307692307693, 
                   0.7836538461538461, 0.7692307692307693]

plt.plot(sequence_lengths, accuracy_scores, marker='o')
# plt.plot(sequence_lengths, accuracy_scores, marker='o', color ='maroon')
plt.title('Accuracy vs Sequence Length (RRC)')
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.savefig("../outputs/images/acc-vs-seq-len-rrc.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.savefig("../outputs/images/acc-vs-seq-len-rrc.png", format="png", bbox_inches="tight", transparent=True)
# plt.show()

number_of_packets = [2,
 7,
 12,
 17,
 22,
 27,
 32,
 37,
 42,
 47,
 52,
 57,
 62,
 67,
 72,
 77,
 82,
 87,
 92,
 97,
 102,
 107,
 112,
 117,
 122,
 127,
 132,
 137,
 142,
 147,
 152,
 157,
 162,
 167,
 172,
 177,
 182,
 187,
 192,
 197,
 202,
 207]

elapsed_time = [0.06532049179077148,
 0.04408860206604004,
 0.042374610900878906,
 0.04621434211730957,
 0.043862342834472656,
 0.046881675720214844,
 0.03836417198181152,
 0.05196380615234375,
 0.04740619659423828,
 0.04849100112915039,
 0.045909881591796875,
 0.04829812049865723,
 0.04878640174865723,
 0.05084538459777832,
 0.0484006404876709,
 0.0453183650970459,
 0.05185818672180176,
 0.047719478607177734,
 0.04923605918884277,
 0.04811382293701172,
 0.05110287666320801,
 0.054767608642578125,
 0.05041670799255371,
 0.0480189323425293,
 0.05043506622314453,
 0.054166316986083984,
 0.05167031288146973,
 0.05647850036621094,
 0.05411958694458008,
 0.0500795841217041,
 0.051737070083618164,
 0.05325627326965332,
 0.05021858215332031,
 0.05077242851257324,
 0.057656288146972656,
 0.054724693298339844,
 0.05798768997192383,
 0.05634760856628418,
 0.053876638412475586,
 0.05525040626525879,
 0.05096912384033203,
 0.05129408836364746]

memory_usage = [
    834.171875,
 834.171875,
 834.171875,
 834.04296875,
 834.04296875,
 834.04296875,
 834.16796875,
 834.29296875,
 834.29296875,
 834.29296875,
 834.29296875,
 834.29296875,
 834.29296875,
 834.41796875,
 834.41796875,
 834.41796875,
 834.28515625,
 834.28515625,
 834.28515625,
 834.28515625,
 834.28515625,
 834.28515625,
 834.28515625,
 834.15625,
 834.15625,
 834.02734375,
 834.02734375,
 834.02734375,
 834.02734375,
 834.15234375,
 834.15234375,
 834.15234375,
 834.15234375,
 834.15234375,
 834.27734375,
 834.14453125,
 834.14453125,
 834.14453125,
 834.01953125,
 834.01953125,
 834.14453125,
 834.01953125
]

power_consumption = [
    0.5878844261169434,
 0.8817720413208008,
 0.9407163619995117,
 0.8965582370758056,
 0.7719772338867188,
 0.7594831466674804,
 0.8401753664016723,
 0.9093666076660156,
 0.9718270301818848,
 0.8922344207763672,
 0.7666950225830078,
 0.782429552078247,
 1.073300838470459,
 0.9914849996566772,
 0.7647301197052002,
 0.9562175035476685,
 1.0112346410751343,
 0.9782493114471436,
 0.9847211837768555,
 0.8852943420410155,
 1.2775719165802002,
 1.3089458465576171,
 1.42679283618927,
 0.9843881130218506,
 1.2003545761108398,
 0.9858269691467285,
 1.2607556343078612,
 1.8468469619750978,
 1.6560593605041505,
 1.0717031002044677,
 1.4124220132827758,
 1.5391062974929808,
 0.8989126205444335,
 1.3556238412857056,
 1.41257905960083,
 1.5979610443115233,
 1.6700454711914063,
 1.724236822128296,
 1.9934356212615967,
 1.4972860097885132,
 0.7645368576049805,
 0.9745876789093018
]

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

plt.plot(number_of_packets, elapsed_time)
plt.title('Time Consumption')
plt.xlabel('Number of Packets')
plt.ylabel('Elapsed Time (ms)')
plt.grid(True)
plt.savefig("../outputs/images/time-vs-packets.pdf", format="pdf", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

plt.plot(number_of_packets, memory_usage)
plt.title('Memory Consumption')
plt.xlabel('Number of Packets')
plt.ylabel('Memory Usage (KB)')
plt.grid(True)
plt.savefig("../outputs/images/memory-vs-packets.pdf", format="pdf", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

plt.plot(number_of_packets, power_consumption)
plt.title('Power Consumption')
plt.xlabel('Number of Packets')
plt.ylabel('Power Consumption (mW)')
plt.grid(True)
plt.savefig("../outputs/images/power-vs-packets.pdf", format="pdf", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Increase the number of data points for smoother interpolation
new_number_of_packets = np.linspace(min(number_of_packets), max(number_of_packets), 300)
spl = make_interp_spline(number_of_packets, elapsed_time, k=3)  # k is the degree of the spline
smooth_elapsed_time = spl(new_number_of_packets)

# Plot the original points
# plt.scatter(number_of_packets, elapsed_time, label='Original Points', color='orange')

# Plot the smoothed graph
plt.plot(new_number_of_packets, smooth_elapsed_time, color='black')

plt.title('Elapsed Time Trend with Spline Interpolation')
plt.xlabel('Number of Packets')
plt.ylabel('Elapsed Time (seconds)')
plt.legend()
plt.grid(True)
plt.savefig("../outputs/images/elapsed-time-trend.pdf", format="pdf", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

# Convert lists to numpy arrays
number_of_packets_np = np.array(number_of_packets).reshape(-1, 1)
elapsed_time_np = np.array(elapsed_time)

# Linear regression
regression_model = LinearRegression().fit(number_of_packets_np, elapsed_time_np)
trend_line = regression_model.predict(number_of_packets_np)

# Increase the number of data points for smoother interpolation
new_number_of_packets = np.linspace(min(number_of_packets), max(number_of_packets), 300)
spl = make_interp_spline(number_of_packets, elapsed_time, k=3)  # k is the degree of the spline
smooth_elapsed_time = spl(new_number_of_packets)

# Plot the original points
plt.scatter(number_of_packets, elapsed_time, label='Original Points', color='green')

# Plot the smoothed graph
plt.plot(new_number_of_packets, smooth_elapsed_time, label='Smoothed Spline')

# Plot the trend line
plt.plot(number_of_packets, trend_line, label='Trend Line', color='green', linestyle='--')

plt.title('Time Consumption')
plt.xlabel('Number of Packets')
plt.ylabel('Detection Time (ms)')
plt.legend()
plt.grid(True)
# plt.savefig("../outputs/images/time-consumption.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.savefig("../outputs/images/time-consumption.png", format="png", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

# Convert lists to numpy arrays
number_of_packets_np = np.array(number_of_packets).reshape(-1, 1)

memory_usage_np = np.array(memory_usage)

# Linear regression
regression_model = LinearRegression().fit(number_of_packets_np, memory_usage_np)
trend_line = regression_model.predict(number_of_packets_np)

# Increase the number of data points for smoother interpolation
new_number_of_packets = np.linspace(min(number_of_packets), max(number_of_packets), 300)
spl = make_interp_spline(number_of_packets, memory_usage, k=3)  # k is the degree of the spline
smooth_memory_usage = spl(new_number_of_packets)

# Plot the original points
plt.scatter(number_of_packets, memory_usage, label='Original Points', color='green')

# Plot the smoothed graph
plt.plot(new_number_of_packets, smooth_memory_usage, label='Smoothed Spline')

# Plot the trend line
plt.plot(number_of_packets, trend_line, label='Trend Line', color='green', linestyle='--')

plt.title('Memory Consumption')
plt.xlabel('Number of Packets')
plt.ylabel('Memory Usage (KB)')
plt.legend()
plt.grid(True)
# plt.savefig("../outputs/images/memory-consumption.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.savefig("../outputs/images/memory-consumption.png", format="png", bbox_inches="tight", transparent=True)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

# Convert lists to numpy arrays
number_of_packets_np = np.array(number_of_packets).reshape(-1, 1)
power_consumption_np = np.array(power_consumption)

# Linear regression
regression_model = LinearRegression().fit(number_of_packets_np, power_consumption_np)
trend_line = regression_model.predict(number_of_packets_np)

# Increase the number of data points for smoother interpolation
new_number_of_packets = np.linspace(min(number_of_packets), max(number_of_packets), 300)
spl = make_interp_spline(number_of_packets, power_consumption, k=3)  # k is the degree of the spline
smooth_power_consumption = spl(new_number_of_packets)

# Plot the original points
plt.scatter(number_of_packets, power_consumption, label='Original Points', color='green')

# Plot the smoothed graph
plt.plot(new_number_of_packets, smooth_power_consumption, label='Smoothed Spline')

# Plot the trend line
plt.plot(number_of_packets, trend_line, label='Trend Line', color='green', linestyle='--')

plt.title('Power Consumption')
plt.xlabel('Number of Packets')
plt.ylabel('Power Consumption (mW)')
plt.legend()
plt.grid(True)
# plt.savefig("../outputs/images/power-consumption.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.savefig("../outputs/images/power-consumption.png", format="png", bbox_inches="tight", transparent=True)
# plt.show()