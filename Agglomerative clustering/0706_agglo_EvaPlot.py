import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, pca
from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import convert_color


from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import label

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
pca_2_spec = pd.DataFrame(pca_2_spec)

pca_3_spec = PCA(n_components = 3).fit_transform(val_spec)
pca_3_spec = pd.DataFrame(pca_3_spec)


linkages = ['single','complete','average']
def agglomerative_eva_plot(linkage,criteria):
    x_list = []
    score_2_list = []
    score_3_list = []
    score_LAB_list = []


    for i in range(2, 200):
        x_list.append(i)
        pred_pca_2_spec = AgglomerativeClustering(linkage=linkage, n_clusters=i).fit_predict(pca_2_spec)
        pred_pca_3_spec = AgglomerativeClustering(linkage=linkage, n_clusters=i).fit_predict(pca_3_spec)
        pred_LAB = AgglomerativeClustering(linkage=linkage, n_clusters=i).fit_predict(val_lab)


        score_2_list.append(round(criteria(val_spec,pred_pca_2_spec),3))
        score_3_list.append(round(criteria(val_spec,pred_pca_3_spec),3))
        score_LAB_list.append(round(criteria(val_spec,pred_LAB),3))
    
    return x_list, score_2_list, score_3_list, score_LAB_list


# x, dbi_2, dbi_3, dbi_lab = agglomerative_eva_plot(linkages[2], davies_bouldin_score)
# plt.plot(x, dbi_2)
# for a, b in zip(x, dbi_2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
#plt.show()

for i in range (0, len(linkages)):      
    x, dbi_2, dbi_3, dbi_lab = agglomerative_eva_plot(linkages[i], davies_bouldin_score)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Evaluation results of agglomerative method in %s linkage'%linkages[i])
    ax1.plot(x, dbi_2, label = '2D')
    ax1.plot(x, dbi_3, label = '3D')
    ax1.plot(x, dbi_lab, label = 'LAB')
    ax1.set_xlabel('num of clusters')
    ax1.set_ylabel('DBI score')
    ax1.legend()


    x, sil_2, sil_3, sil_lab = agglomerative_eva_plot(linkages[i], silhouette_score)
    ax2.plot(x, sil_2, label = '2D')
    ax2.plot(x, sil_3, label = '3D')
    ax2.plot(x, sil_lab, label = 'LAB')
    ax2.set_xlabel('num of clusters')
    ax2.set_ylabel('SIL score')
    ax2.legend()
    plt.show()


