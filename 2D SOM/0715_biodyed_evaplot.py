from networkx.algorithms import cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
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



'''
fig, (ax1,ax2) = plt.subplots(1, 2)
fig.suptitle('Clustering result of 2DSOM')

ax1.plot(x, dbi_list, label = 'DBI', c = 'b')
#ax1.plot(x, quan_err_list, label = 'quantization error')
ax1.set_xlabel('number of neurons')
ax1.set_ylabel('DBI score')
ax1.legend()


ax2.plot(x, sil_list, label = 'SIL', c ='g')
#ax1.plot(x, quan_err_list, label = 'quantization error')
ax2.set_xlabel('number of neurons')
ax2.set_ylabel('SIL score')
ax2.legend()
plt.show()
'''


fig, ax1 = plt.subplots(1, 1)

ax1.plot(x, quan_err_list, c = 'blue')
ax1.scatter(x, quan_err_list, marker = 'o', c ='blue', s = 10)
ax1.set_xlabel('number of neurons')
ax1.set_ylabel('quantization error')
# for i in range(len(x)):
#     ax1.text(x[i], quan_err_list[i], "("+str(round(x[i],0))+', '+str(round(quan_err_list[i], 3))+")", fontsize = 7)
plt.title('Quantization error at different map size of 2DSOM')
plt.show()
print(quan_err_list)


'''

fig, ax = plt.subplots()
#clusters = []
texts = []
for ix in range(val_spec_norm.shape[0]):
    cluster = som.winner(val_spec_norm[ix])
    plt.scatter(x = cluster[0], y = cluster[1], alpha = 1, c = np.array([val_rgb[ix, :]/255]), s = 120)
    cl_x = cluster[0] - 0.4
    cl_y = cluster[1]
    if (cl_x, cl_y) not in texts:
        texts.append((cl_x, cl_y))
        ax.annotate(ix, (cl_x, cl_y), fontsize = 7)
    else:
        cnt = 0
        while True:
            
            cl_x = cl_x + 0.18
            cnt += 1
            if cnt > 4:
                cl_x = cl_x - 0.18 * 5
                cl_y = cl_y - 0.35
                cnt = 0
                
            if (cl_x, cl_y) not in texts:
                break
        
        ax.annotate(ix, (cl_x, cl_y), fontsize = 7)
        texts.append((cl_x, cl_y))



        


    
#print(texts)
#print(np.shape(val_spec_norm))
#adjust_text(texts)


    

ax.set_title('Cluster results by 2D SOM')  
plt.xlabel('X.neurons')   
plt.ylabel('Y.neurons')  
#plt.ylim((0, 30))
#plt.xlim((0, 20))
plt.show()




#sc = plt.scatter(x = clusters[:,1], y = clusters[:,0], alpha = 1, c = val_rgb[:]/255)
#for cnt, c in enumerate(clusters):
   
#    plt.scatter(x = c[1], y = c[0], alpha = 1, c = np.array([val_rgb[cnt, :]/255]))
'''