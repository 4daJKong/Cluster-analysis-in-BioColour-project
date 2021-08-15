
import numpy as np
import pandas as pd
from GHSOM import GHSOM
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize
from neuron.neuron import Neuron
from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import convert_color


data_shape = 1


def __gmap_to_matrix(gmap):
    gmap = gmap[0]
    map_row = data_shape * gmap.shape[0]
    map_col = data_shape * gmap.shape[1]
    _image = np.empty(shape=(map_row, map_col), dtype=np.float32)
    for i in range(0, map_row, data_shape):
        for j in range(0, map_col, data_shape):
            neuron = gmap[i // data_shape, j // data_shape]
            _image[i:(i + data_shape), j:(j + data_shape)] = np.reshape(neuron, newshape=(data_shape, data_shape))
    return _image


def __plot_child(e, gmap, level):
    if e.inaxes is not None:
        coords = (int(e.ydata // data_shape), int(e.xdata // data_shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_plot(neuron.child_map, num=str(coords), level=level+1)


def interactive_plot(gmap, num='root', level=1):
    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)
    ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event', lambda event: __plot_child(event, gmap, level))
    plt.axis('off')
    fig.show()


def __plot_child_with_labels(e, gmap, level, data, labels, colours, associations):
    if e.inaxes is not None:
        coords = (int(e.xdata), int(e.ydata))
        #print(coords)
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            assc = associations[coords[0]][coords[1]]
            #print("assc num:", len(assc))
            interactive_plot_with_labels(neuron.child_map, dataset=data[assc], labels=labels[assc], colours=colours[assc],
                                         num=str(coords), level=level+1)


def interactive_plot_with_labels(gmap, dataset, labels, colours, num='root', level=1):


    mapping = [[list() for _ in range(gmap.map_shape()[1])] for _ in range(gmap.map_shape()[0])]

    _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots(num=_num)


    #ax.imshow(__gmap_to_matrix(gmap.weights_map), cmap='bone_r', interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event', lambda event: __plot_child_with_labels(event, gmap, level, dataset, labels,colours, mapping))
    

    cluster = []
    for idx in range(labels.shape[0]):
        winner_neuron = gmap.winner_neuron(dataset[idx])[0][0]
        r, c = winner_neuron.position
        #print(r, c)
        mapping[r][c].append(idx)
        cl_x = r + np.random.rand() * 0.75
        cl_y = c + np.random.rand() * 0.75
        cluster.append([idx, cl_x , cl_y])
        if level != 1: 
            ax.annotate(labels[idx], (cl_x, cl_y), fontsize = 7)
    cluster = np.array(cluster)
    sc = plt.scatter(x = cluster[:,1] , y = cluster[:,2], c = colours[:]/255)
    plt.plot((0, 1.75), (0.875, 0.875), ":",c = '#AAAAAA')
    plt.plot((0.875, 0.875),  (0, 1.75), ":", c = '#AAAAAA')
    plt.yticks([0.375,1.375], ['0','1'])  
    plt.xticks([0.375,1.375], ['0','1'])  
  
    
    ax.set_title('GHSOM_layer:'+str(level))  
    ax.set_xlabel('X.neurons')   
    ax.set_ylabel('Y.neurons')  
    fig.show()
    

def mean_data_centroid_activation(ghsom, dataset):
    distances = list()

    for data in dataset:
        _neuron = ghsom
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]
        distances.append(_neuron.activation(data))

    distances = np.asarray(a=distances, dtype=np.float32)
    return distances.mean(), distances.std()


def __number_of_neurons(root):
    r, c = root.child_map.weights_map[0].shape[0:2]
    total_neurons = r * c
    for neuron in root.child_map.neurons.values():
        if neuron.child_map is not None:
            total_neurons += __number_of_neurons(neuron)
    return total_neurons


def dispersion_rate(ghsom, dataset):
    used_neurons = dict()
    for data in dataset:
        gsom_reference = ''
        neuron_reference = ''
        _neuron = ghsom
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]

            gsom_reference = str(_gsom)
            neuron_reference = str(_neuron)

        used_neurons["{}-{}-{}".format(gsom_reference, neuron_reference, _neuron.position)] = True
    used_neurons = len(used_neurons)

    return __number_of_neurons(ghsom) / used_neurons





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

if __name__ == '__main__':


    df = pd.read_excel('0419_combined_biodyes.xlsx', header=0)
    df_name = df.values[0:,1]
    val_idx = df.values[0:, 0]
    val_lab = df.values[0:, 2:5]
    val_spec = df.values[0:,5:]
    val_rgb = Conv_lab_rgb(val_lab)
    
    val_spec_norm = normalize(val_spec)
    data =  np.array(val_spec_norm, dtype = np.float64)

   


    ghsom = GHSOM(input_dataset=data, t1=0.8, t2=0.08, learning_rate=0.15, decay=0.95, gaussian_sigma=1.5)

    print("Training...")
    zero_unit = ghsom.train(epochs_number=10, dataset_percentage=0.50, min_dataset_size=10, grow_maxiter=5)
    print("zero_unit",zero_unit)
    print("the number of nueron units:",Neuron.cnt_num_times)

    
    interactive_plot_with_labels(zero_unit.child_map, data, val_idx, val_rgb)
    
    plt.show()










