import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import Counter
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import Lab_to_LCHab, convert_color

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


pca_2_spec = PCA(n_components = 2).fit_transform(val_spec)
pca_3_spec = PCA(n_components = 3).fit_transform(val_spec)



pred_pca_2_spec = OPTICS(min_samples= 4, eps= 10, cluster_method = 'dbscan').fit_predict(pca_2_spec)
pred_pca_3_spec = OPTICS(min_samples= 4, eps= 10, cluster_method = 'dbscan').fit_predict(pca_3_spec)
pred_LAB = OPTICS(min_samples= 4, eps= 10, cluster_method = 'dbscan').fit_predict(val_lab)

'''
#2d pca
fig, ax = plt.subplots()
ax.set_title('Clusters by OPTICS in 2D space after PCA')
ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')

points = ax.scatter(
    pca_2_spec[:,0], 
    pca_2_spec[:,1],
    s = 7,
    marker='o',
    #label = pred_pca_2_spec[i],
    c = pred_pca_2_spec,
    cmap= 'tab20')

points_noise = ax.scatter(pca_2_spec[np.where(pred_pca_2_spec == -1),0], 
        pca_2_spec[np.where(pred_pca_2_spec == -1),1], 
                c = 'k', s = 7, marker='o')

handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_pca_2_spec).items()])
one_more = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize = handles[0].get_ms())
ax.legend([one_more] + handles[1:], labels, loc = "lower right",title = 'clusters')  
plt.show()
'''

'''
#3d PCA only one figure
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_title('Clusters by OPTICS in 3D space after PCA')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
points = ax.scatter3D(
        pca_3_spec[:,0], 
        pca_3_spec[:,1],
        pca_3_spec[:,2],
        s = 8,
        marker='o',
        c = pred_pca_3_spec
    )
ax.legend(*points.legend_elements(), title = 'clusters')  
plt.show()
'''

'''
#3D LAB
fig = plt.figure()

ax = plt.axes(projection = '3d')
ax.set_title('Clusters by OPTICS in LAB color space')
ax.set_xlabel('L')
ax.set_ylabel('A')
ax.set_zlabel('B')
val_lab_x, val_lab_y,val_lab_z = val_lab[:,0], val_lab[:,1], val_lab[:,2]

points = ax.scatter3D(
        [float(i) for i in val_lab_x], 
        [float(i) for i in val_lab_y],
        [float(i) for i in val_lab_z],
        #val_lab[:,1],
        #val_lab[:,2],
        s = 8,
        marker='o',
        c = pred_LAB,
        #c = reduce(operator.add, pred_LAB)
        cmap = 'rainbow'
    )

ax.legend(*points.legend_elements(), title = 'clusters')  
plt.show()
'''



'''
#3d PCA with 2D projection
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(pca_3_spec[:,0], 
        pca_3_spec[:,1],
        s = 5,
        marker='.',
        c = pred_pca_3_spec,
        cmap = 'tab20')
ax1.set_ylabel('Second Component')

ax2 = fig.add_subplot(2,2,2, projection = '3d')
ax2.set_xlabel('First Component')
ax2.set_ylabel('Second Component')
ax2.set_zlabel('Third Component')

points = ax2.scatter3D(
        pca_3_spec[:,0], 
        pca_3_spec[:,1],
        pca_3_spec[:,2],
        s = 5,
        marker='.',
        c = pred_pca_3_spec,
        cmap = 'tab20'
    )

ax3 = fig.add_subplot(2,2,3)
ax3.scatter(pca_3_spec[:,0], 
        pca_3_spec[:,2],
        s = 5,
        marker='.',
        c = pred_pca_3_spec,
        cmap = 'tab20')
ax3.set_xlabel('First Component')
ax3.set_ylabel('Third Component')


ax4 = fig.add_subplot(2,2,4)
ax4.scatter(pca_3_spec[:,1], 
        pca_3_spec[:,2],
        s = 5,
        marker='.',
        c = pred_pca_3_spec,
        cmap = 'tab20')
ax4.set_xlabel('Second Component')

points_noise_1 = ax1.scatter(
        pca_3_spec[np.where(pred_pca_3_spec == -1),0], 
        pca_3_spec[np.where(pred_pca_3_spec == -1),1],
                c = 'k', s = 5, marker='.')
points_noise_2 = ax2.scatter(
        pca_3_spec[np.where(pred_pca_3_spec == -1),0], 
        pca_3_spec[np.where(pred_pca_3_spec == -1),1],
        pca_3_spec[np.where(pred_pca_3_spec == -1),2], 
                c = 'k', s = 5, marker='.')
points_noise_3 = ax3.scatter(
        pca_3_spec[np.where(pred_pca_3_spec == -1),0], 
        pca_3_spec[np.where(pred_pca_3_spec == -1),2],
                c = 'k', s = 5, marker='.')
points_noise_4 = ax4.scatter(
        pca_3_spec[np.where(pred_pca_3_spec == -1),1], 
        pca_3_spec[np.where(pred_pca_3_spec == -1),2],
                c = 'k', s = 5, marker='.')


labels = ['-1: 161', '0: 368','1: 3', '2: 4', '3: 4', '4: 9', '5: 3', '6: 6', '7: 8', '8: 9', '9: 8', '10: 4', '11: 4', '12: 4', '13: 6', '14: 4', '15: 4', '16: 2']
handles, _ = points.legend_elements(num = len(labels))

one_more = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize = handles[0].get_ms())

fig.legend([one_more] + handles[1:], labels, loc = "upper right",title = 'clusters')  
fig.suptitle('Clusters by OPTICS in 3D space after PCA')
plt.show()
'''


#3d LAB with 2D projection
df_lab = pd.DataFrame(val_lab, columns=['L','A','B'])

val_lab_x = []
val_lab_y = []
val_lab_z = []
for i in range(len(val_lab)):
        if pred_LAB[i] != -1:
                val_lab_x.append(float(val_lab[i,0])) 
                val_lab_y.append(float(val_lab[i,1])) 
                val_lab_z.append(float(val_lab[i,2])) 

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(
        df_lab[pred_LAB != -1].values[:,0], 
        df_lab[pred_LAB != -1].values[:,1],
        s = 5,
        marker='.',
        c = pred_LAB[pred_LAB != -1],
        cmap = 'tab20')
ax1.set_ylabel('A')

ax2 = fig.add_subplot(2,2,2, projection = '3d')
ax2.set_xlabel('L')
ax2.set_ylabel('A')
ax2.set_zlabel('B')
points = ax2.scatter3D(
        val_lab_x, 
        val_lab_y,
        val_lab_z,
        s = 5,
        marker='.',
        #c = pred_LAB,
        c = pred_LAB[pred_LAB != -1],
        cmap = 'tab20'
    )

ax3 = fig.add_subplot(2,2,3)
ax3.scatter(
        df_lab[pred_LAB != -1].values[:,0], 
        df_lab[pred_LAB != -1].values[:,2],
        s = 5,
        marker='.',
        c = pred_LAB[pred_LAB != -1],
        cmap = 'tab20')
ax3.set_xlabel('L')
ax3.set_ylabel('B')

ax4 = fig.add_subplot(2,2,4)
ax4.scatter(
        # df_lab.values[:,1], 
        # df_lab.values[:,2],
        df_lab[pred_LAB != -1].values[:,1], 
        df_lab[pred_LAB != -1].values[:,2],
        s = 5,
        marker='.',
        c = pred_LAB[pred_LAB != -1],
        cmap = 'tab20')
ax4.set_xlabel('A')
#print([float(i) for i in val_lab_x])

points_noise_1 = ax1.scatter(

        df_lab[pred_LAB == -1].values[:,0],
        df_lab[pred_LAB == -1].values[:,1],
                c = 'k', s = 5, marker='.')
                
for i in range(0, len(val_lab)):
        if pred_LAB[i] == -1:
                ax2.scatter(
                        val_lab_x[i],
                        val_lab_y[i],
                        val_lab_z[i],
                        c = 'k', s = 5, marker='o'
                )

points_noise_3 = ax3.scatter(
        df_lab[pred_LAB == -1].values[:,0],
        df_lab[pred_LAB == -1].values[:,2],
                c = 'k', s = 5, marker='.')

points_noise_4 = ax4.scatter(
        df_lab[pred_LAB == -1].values[:,1],
        df_lab[pred_LAB == -1].values[:,2],
                c = 'k', s = 5, marker='.')

handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_LAB).items()])
one_more = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize = handles[0].get_ms())
fig.legend([one_more] + handles, labels, loc = "lower right",title = 'clusters')  
fig.suptitle('Clusters by OPTICS in LAB color space')
plt.show()

