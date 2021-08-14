
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import convert_color

from sklearn.preprocessing import normalize
from minisom import MiniSom
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
def Conv_lab_rgb (Value_lab):
    val_lab[Value_lab[:,0] > 100] = 100 
    Value_rgb = []
    for lab_list in Value_lab:
       lab = LabColor(*[component for component in lab_list])
       rgb = convert_color(lab, sRGBColor)
       rgb_list = [255*color for color in rgb.get_value_tuple()]
       Value_rgb.append(rgb_list)
    Value_rgb = np.round(Value_rgb).astype('uint8')
    Value_rgb[Value_rgb <= 0] = 0
    Value_rgb[Value_rgb >= 255] = 255
    return Value_rgb


df = pd.read_excel('0419_combined_biodyes.xlsx', header=0)

df_name = df.values[0:, 1]
val_idx = df.values[0:, 0]
val_lab = df.values[0:, 2:5]
val_spec = df.values[0:,5:]
val_rgb = Conv_lab_rgb(val_lab)


val_spec_norm = normalize(val_spec)

quan_err_list = []
x = []
dbi_list = []
sil_list = []

n_neurons = [1,7,9,10,10,10,10,12,14,16,18,20,20,20]
m_neurons = [2,8,9,12,14,16,20,20,20,20,20,20,25,30]
for i in range(0, len(n_neurons)):
    som = MiniSom(n_neurons[i], m_neurons[i], val_spec_norm.shape[1], sigma=1.5, learning_rate=0.5, 
                    neighborhood_function='gaussian', 
                    random_seed=0)
    som.train(val_spec_norm, 500, verbose=True)

    cluster = []
    for ix in range(val_spec_norm.shape[0]):
        cluster.append(som.winner(val_spec_norm[ix]))
    label_list = [list(set(cluster)).index(i) for i in cluster]
    dbi_list.append(davies_bouldin_score(val_spec, label_list))
    sil_list.append(silhouette_score(val_spec,label_list))

    quan_err_list.append(som.quantization_error(val_spec_norm))
    x.append(n_neurons[i] * m_neurons[i])

fig, ax1 = plt.subplots(1, 1)

ax1.plot(x, quan_err_list, c = 'blue')
ax1.scatter(x, quan_err_list, marker = 'o', c ='blue', s = 10)
ax1.set_xlabel('number of neurons')
ax1.set_ylabel('quantization error')

plt.title('Quantization error at different map size of 2DSOM')
plt.show()


