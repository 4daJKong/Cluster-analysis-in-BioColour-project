import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import convert_color

from sklearn.preprocessing import normalize
from minisom import MiniSom

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


df = pd.read_excel('combined_biodyes.xlsx', header=0)

df_name = df.values[0:, 1]
val_idx = df.values[0:, 0]
val_lab = df.values[0:, 2:5]
val_spec = df.values[0:,5:]
val_rgb = Conv_lab_rgb(val_lab)


val_spec_norm = normalize(val_spec)

som = MiniSom(14, 10, val_spec_norm.shape[1], sigma=1.5, learning_rate=0.5, 
                neighborhood_function='gaussian', 
                random_seed=0)

som.train(val_spec_norm, 500, verbose=True)



fig, ax = plt.subplots()
#clusters = []
texts = []
for ix in range(val_spec_norm.shape[0]):
    cluster = som.winner(val_spec_norm[ix])
    plt.scatter(x = cluster[0], y = cluster[1], alpha = 1, c = np.array([val_rgb[ix, :]/255]), s = 100)
    cl_x = cluster[0] - 0.4
    cl_y = cluster[1] - 0.4

    
    cnt = 0
    while True:
        if (cl_x, cl_y) not in texts:
            texts.append((cl_x, cl_y))
            ax.annotate(ix, (cl_x, cl_y), fontsize = 7)
            break
        else:
            cl_x = cl_x + 0.23
            cnt = cnt + 1 
            if cnt > 2:
                texts.append((cl_x, cl_y))
                ax.annotate('...', (cl_x, cl_y), fontsize = 7)
                break
                
ax.set_title('Clustering result by 2D SOM')  
plt.xlabel('X.neurons')   
plt.ylabel('Y.neurons')  
plt.show()
