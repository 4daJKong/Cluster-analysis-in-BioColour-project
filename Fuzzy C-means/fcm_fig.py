import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from fcmeans import FCM

from sklearn.decomposition import PCA, pca


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


df = pd.read_excel('dataset/0419_combined_biodyes.xlsx', header=0)



df_name = df.values[0:, 1]
val_idx = df.values[0:, 0]
val_lab = df.values[0:, 2:5]
val_spec = df.values[0:,5:]
val_rgb = Conv_lab_rgb(val_lab)


pca_2_spec = PCA(n_components = 2).fit_transform(val_spec)
pca_3_spec = PCA(n_components = 3).fit_transform(val_spec)



pred_pca_2_spec = FCM(m = 1.2, n_clusters=8).fit(pca_2_spec).predict(pca_2_spec)
pred_pca_3_spec = FCM(m = 1.2, n_clusters=8).fit(pca_3_spec).predict(pca_3_spec)
pred_LAB = FCM(m = 1.2, n_clusters=8).fit(val_lab).predict(val_lab)

'''
#2d PCA
fig, ax = plt.subplots()
ax.set_title('Clusters by FCM in 2D space after PCA')
ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')
points = ax.scatter(
    pca_2_spec[:,0], 
    pca_2_spec[:,1],
    s = 7,
    marker='o',
    c = pred_pca_2_spec,
    cmap= 'Dark2')
# for i in range(0, len(df_name)):
#     ax.annotate(val_idx[i], tuple(pca_2_spec.values[i]), ha = "center", fontsize = 4)
#plt.savefig('aggglo_pca.png', dpi = 200) 
handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_pca_2_spec).items()])
ax.legend(handles, labels, loc = "lower right",title = 'clusters')  
plt.show()
'''

'''
#3d PCA
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_title('Clusters by FCM in 3D space after PCA')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
points = ax.scatter3D(
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
ax.legend(handles, labels, loc = "lower right",title = 'clusters')  
plt.show()
'''


'''
#3d lab
fig = plt.figure()
#ax = fig.add_subplot(projection = '3d')

ax = plt.axes(projection = '3d')
ax.set_title('Clusters by FCM in LAB color space')
ax.set_xlabel('L')
ax.set_ylabel('A')
ax.set_zlabel('B')
val_lab_x, val_lab_y,val_lab_z = val_lab[:,0], val_lab[:,1], val_lab[:,2]
#for i in range(0, len(df_name)):
#for i in range(0, len(df_name)):
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
        cmap = 'Dark2'
    )
handles, _ = points.legend_elements()
labels =sorted([f'{item}: {count}' for item, count in Counter(pred_LAB).items()])
ax.legend(handles, labels, loc = "lower right",title = 'clusters')  
plt.show()
'''

'''
#subplots
df_val_pca_3 = pd.DataFrame({'1st component':pca_3_spec[:,0].astype(np.float),
     '2nd component':pca_3_spec[:,1].astype(np.float),
     '3rd component':pca_3_spec[:,2].astype(np.float)})


df_val_lab = pd.DataFrame({'L':val_lab[:,0].astype(np.float),
     'A':val_lab[:,1].astype(np.float),
     'B':val_lab[:,2].astype(np.float)})

fig = pd.plotting.scatter_matrix(
    df_val_lab,  
    #df_val_pca_3,
    s = 15,
    marker = '.',
    
    c = pred_LAB,
    #c = pred_pca_3_spec,
    
    cmap = 'Dark2'
    )
handles = [plt.plot([],[], ls="", marker="o", \
                    markersize=np.sqrt(16))[0] for i in range(8)]
labels = sorted([f'{item}: {count}' for item, count in Counter(pred_LAB).items()])
plt.legend(handles, labels, loc=(1.02,0), title = 'clusters')

plt.suptitle('Clusters by FCM in LAB space')
#plt.suptitle('Clusters by FCM in 3D space after PCA')
#plt.savefig('C:/Users/WinstonLi/Desktop/0707_clustermethod\PCA_pair.png', dpi = 250) 
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
fig.suptitle('Clusters by FCM in 3D space after PCA')


plt.show()
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
fig.suptitle('Clusters by FCM in LAB color space')


plt.show()
