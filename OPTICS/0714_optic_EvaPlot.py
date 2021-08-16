import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import OPTICS

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
#pca_2_spec = pd.DataFrame(pca_2_spec)
#print(np.linalg.norm(pca_2_spec[0,:] - pca_2_spec[1,:]))
pca_3_spec = PCA(n_components = 3).fit_transform(val_spec)
#pca_3_spec = pd.DataFrame(pca_3_spec)





def FCM_eva_plot(criteria):
    x_list = []
    score_2_list = []
    score_3_list = []
    score_LAB_list = []

    num_clu_2 = []
    num_clu_3 = []
    num_clu_LAB = []
    for i in np.arange(2, 12, 1):  #minpts
    #for i in np.arange(4, 28, 1): #eps
        x_list.append(i)

        #Minpts:
        pred_pca_2_spec = OPTICS(min_samples= i, max_eps= 10, cluster_method = 'dbscan').fit_predict(pca_2_spec)
        pred_pca_3_spec = OPTICS(min_samples= i, max_eps= 10, cluster_method = 'dbscan').fit_predict(pca_3_spec)
        pred_LAB = OPTICS(min_samples= i, max_eps= 10, cluster_method = 'dbscan').fit_predict(val_lab)

        #eps:
        # pred_pca_2_spec = OPTICS(min_samples= 4, eps= i, cluster_method = 'dbscan').fit_predict(pca_2_spec)
        # pred_pca_3_spec = OPTICS(min_samples= 4, eps= i, cluster_method = 'dbscan').fit_predict(pca_3_spec)
        # pred_LAB = OPTICS(min_samples= 4, eps= i, cluster_method = 'dbscan').fit_predict(val_lab)

        # num_clu_2.append(np.count_nonzero(np.unique(pred_pca_2_spec)))
        # num_clu_3.append(np.count_nonzero(np.unique(pred_pca_3_spec)))
        # num_clu_LAB.append(np.count_nonzero(np.unique(pred_LAB)))

        score_2_list.append(round(criteria(val_spec,pred_pca_2_spec),3))
        score_3_list.append(round(criteria(val_spec,pred_pca_3_spec),3))
        score_LAB_list.append(round(criteria(val_spec,pred_LAB),3))
    
    #return x_list, num_clu_2, num_clu_3, num_clu_LAB
    return x_list, score_2_list, score_3_list, score_LAB_list


#x, dbi_2, dbi_3, dbi_lab = FCM_eva_plot(mem[0], davies_bouldin_score)
#print(FCM(m = 1.1, n_clusters= 8).fit(pca_2_spec).predict(pca_2_spec))
#plt.plot(x, dbi_2)
# for a, b in zip(x, dbi_2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
#plt.show()


'''
x, dbi_2, dbi_3, dbi_lab = FCM_eva_plot(davies_bouldin_score)
fig, ax1 = plt.subplots()
ax1.set_title('Clustering result of OPTICS')
ax1.bar([i - 0.25 for i in x], dbi_2, label = '2D', width = 0.25)
ax1.bar(x, dbi_3, label = '3D',  width = 0.25)
ax1.bar([i + 0.25 for i in x], dbi_lab, label = 'LAB',  width = 0.25)
ax1.set_xlabel('Minpts:minimum number of points in ε-neighborhood')
ax1.set_ylabel('corresponding number of clusters')
# for i in range(0, len(x)):
#     ax1.text(x[i] - 0.4, dbi_2[i], str(dbi_2[i]), fontsize = 8)
#     ax1.text(x[i] - 0.15, dbi_3[i], str(dbi_3[i]), fontsize = 8)
#     ax1.text(x[i] + 0.1, dbi_lab[i], str(dbi_lab[i]), fontsize = 8)
for i in range(len(x)):
    plt.annotate(f'{dbi_2[i]}\n', xy=(x[i] - 0.25, dbi_2[i]), ha='center', va='center', fontsize = 9)
    plt.annotate(f'{dbi_3[i]}\n', xy=(x[i] , dbi_3[i]), ha='center', va='center', fontsize = 9)
    plt.annotate(f'{dbi_lab[i]}\n', xy=(x[i] + 0.25, dbi_lab[i]), ha='center', va='center', fontsize = 9)
ax1.legend()
plt.xticks(x)
plt.show()   
'''


#x:Minpts
x, dbi_2, dbi_3, dbi_lab = FCM_eva_plot(davies_bouldin_score)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Evaluation results of OPTICS in Eps = 10')
ax1.plot(x, dbi_2, label = '2D')
ax1.plot(x, dbi_3, label = '3D')
ax1.plot(x, dbi_lab, label = 'LAB')
ax1.set_xlabel('Minpts:minimum number of points in ε-neighborhood')
ax1.set_ylabel('DBI score')
ax1.legend()


x, sil_2, sil_3, sil_lab = FCM_eva_plot(silhouette_score)
ax2.plot(x, sil_2, label = '2D')
ax2.plot(x, sil_3, label = '3D')
ax2.plot(x, sil_lab, label = 'LAB')
ax2.set_xlabel('Minpts:minimum number of points in ε-neighborhood')
ax2.set_ylabel('SIL score')
ax2.legend()
plt.show()


'''
#x: eps
x, dbi_2, dbi_3, dbi_lab = FCM_eva_plot(davies_bouldin_score)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Evaluation results of OPTICS in Minpts = 4')
ax1.plot(x, dbi_2, label = '2D')
ax1.plot(x, dbi_3, label = '3D')
ax1.plot(x, dbi_lab, label = 'LAB')
ax1.set_xlabel('Eps: the size of ε-neighborhood about the radius')
ax1.set_ylabel('DBI score')
ax1.legend()


x, sil_2, sil_3, sil_lab = FCM_eva_plot(silhouette_score)
ax2.plot(x, sil_2, label = '2D')
ax2.plot(x, sil_3, label = '3D')
ax2.plot(x, sil_lab, label = 'LAB')
ax2.set_xlabel('Eps: the size of ε-neighborhood about the radius')
ax2.set_ylabel('SIL score')
ax2.legend()
plt.show()
'''