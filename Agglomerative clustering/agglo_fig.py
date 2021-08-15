import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, pca

from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import Lab_to_LCHab, convert_color

from collections import Counter

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
val_lab = np.array(df.values[0:, 2:5])
val_spec = df.values[0:,5:]
val_rgb = Conv_lab_rgb(val_lab)


pca_2_spec = PCA(n_components = 2).fit_transform(val_spec)
pca_2_spec = pd.DataFrame(pca_2_spec)
pca_3_spec = PCA(n_components = 3).fit_transform(val_spec)


linkages = ['single','complete','average']
pred_pca_2_spec = AgglomerativeClustering(linkage=linkages[2], n_clusters = 8).fit_predict(pca_2_spec)
pred_pca_3_spec = AgglomerativeClustering(linkage=linkages[2], n_clusters = 8).fit_predict(pca_3_spec)
pred_LAB = AgglomerativeClustering(linkage=linkages[2], n_clusters = 8).fit_predict(val_lab)


#2d PCA
fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('Agglomerative hierarchical clusters in 2D space after PCA')
ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')
points = ax.scatter(pca_2_spec.values[:,0], 
    pca_2_spec.values[:,1],
    s = 7,
    marker='o',
    c = pred_pca_2_spec,
    cmap= 'Dark2')
handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_pca_2_spec).items()])
ax.legend(handles, labels, loc = "lower right",title = 'clusters') 
plt.show()


'''
#3d PCA
fig = plt.figure()
ax1 = plt.axes(projection = '3d')
ax1.set_title('Agglomerative hierarchical clusters in 3D space after PCA')
ax1.set_xlabel('First Component')
ax1.set_ylabel('Second Component')
ax1.set_zlabel('Third Component')

points = ax1.scatter3D(
        pca_3_spec[:,0], 
        pca_3_spec[:,1],
        pca_3_spec[:,2],
        s = 8,
        marker='o',
        c = pred_pca_3_spec,
        cmap = 'Dark2'
    )

handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_pca_3_spec).items()])
ax1.legend(handles, labels, loc = "lower right",title = 'clusters')  
plt.show()
'''

'''
#3d LAB
ax = plt.axes(projection = '3d')
ax.set_title('Agglomerative Hierarchical Clusters in LAB color space')
ax.set_xlabel('L')
ax.set_ylabel('A')
ax.set_zlabel('B')
val_lab_x, val_lab_y,val_lab_z = val_lab[:,0], val_lab[:,1], val_lab[:,2]

points = ax.scatter3D(
        [float(i) for i in val_lab_x], 
        [float(i) for i in val_lab_y],
        [float(i) for i in val_lab_z],
        s = 8,
        marker='o',
        c = pred_LAB,
      
        cmap = 'Dark2'
    )
handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_LAB).items()])
ax.legend(handles, labels, loc = "lower right",title = 'clusters')  
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
        cmap = 'Dark2')
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
        cmap = 'Dark2'
    )

ax3 = fig.add_subplot(2,2,3)
ax3.scatter(pca_3_spec[:,0], 
        pca_3_spec[:,2],
        s = 5,
        marker='.',
        c = pred_pca_3_spec,
        cmap = 'Dark2')
ax3.set_xlabel('First Component')
ax3.set_ylabel('Third Component')


ax4 = fig.add_subplot(2,2,4)
ax4.scatter(pca_3_spec[:,1], 
        pca_3_spec[:,2],
        s = 5,
        marker='.',
        c = pred_pca_3_spec,
        cmap = 'Dark2')
ax4.set_xlabel('Second Component')


handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_pca_3_spec).items()])
fig.legend(handles, labels, loc = "lower right",title = 'clusters')  
fig.suptitle('Agglomerative hierarchical clusters in 3D space after PCA')
plt.show()
'''


'''
#3d LAB with 2D projection
val_lab_x, val_lab_y,val_lab_z = val_lab[:,0], val_lab[:,1], val_lab[:,2]
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(val_lab_x, 
        val_lab_y,
        s = 5,
        marker='.',
        c = pred_LAB,
        cmap = 'Dark2')
ax1.set_ylabel('A')

ax2 = fig.add_subplot(2,2,2, projection = '3d')
ax2.set_xlabel('L')
ax2.set_ylabel('A')
ax2.set_zlabel('B')

points = ax2.scatter3D(
        [float(i) for i in val_lab_x], 
        [float(i) for i in val_lab_y],
        [float(i) for i in val_lab_z],
        s = 5,
        marker='.',
        c = pred_LAB,
        cmap = 'Dark2'
    )

ax3 = fig.add_subplot(2,2,3)
ax3.scatter(val_lab_x, 
        val_lab_z,
        s = 5,
        marker='.',
        c = pred_LAB,
        cmap = 'Dark2')
ax3.set_xlabel('L')
ax3.set_ylabel('B')

ax4 = fig.add_subplot(2,2,4)
ax4.scatter(val_lab_y, 
        val_lab_z,
        s = 5,
        marker='.',
        c = pred_LAB,
        cmap = 'Dark2')
ax4.set_xlabel('A')

handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_LAB).items()])
fig.legend(handles, labels, loc = "lower right",title = 'clusters')  
fig.suptitle('Agglomerative hierarchical clusters in LAB color space')

plt.show()
'''
