# %%
# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.colors import ListedColormap
import time
import json
import os, argparse
import tensorflow as tf
from tensorflow.keras import layers, Model
import scipy.io as sio
import seaborn as sns
from scipy.signal import medfilt
import pandas as pd
from operator import truediv
import spectral
import spectral.io.envi as envi
from sklearn.decomposition import (IncrementalPCA, KernelPCA, PCA, SparsePCA,
                                   TruncatedSVD, FactorAnalysis)
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, Conv2D, BatchNormalization, GlobalAveragePooling2D,Flatten,MaxPooling2D
from tensorflow.keras.layers import LayerNormalization
import keras
from tensorflow.keras import layers
from keras_cv_attention_models import attention_layers
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import legacy
from keras import backend as Kb
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, Lambda, multiply)
from tensorflow.keras.optimizers import Adam, AdamW,SGD
from keras.optimizers import Adam
from operator import truediv
import numpy as np
from typing import List
# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))
tf.keras.backend.clear_session()


# %%
base_dir = './datasets'
def LoadHSIData(method):
    data_path = os.path.join(os.getcwd(),f'{base_dir}')
    if method == 'LK':
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt'))['WHU_Hi_LongKou_gt']
        Num_Classes = 9
        target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                        'Narrow-leaf soybean', 'Rice', 'Water',
                        'Roads and houses', 'Mixed weed']
        
    elif method == 'HH':
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt'))['WHU_Hi_HongHu_gt']
        Num_Classes = 22
        target_names = ['Red roof', 'Road', 'Bare soil', 'Cotton',
                        'Cotton firewood', 'Rape', 'Chinese cabbage',
                        'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis',
                        'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa',
                        'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
                        'Carrot', 'White radish', 'Garlic sprout', 'Broad bean',
                        'Tree']
        
    elif method == 'HC':
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt'))['WHU_Hi_HanChuan_gt']
        Num_Classes = 16
        target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum',
                        'Water spinach', 'Watermelon', 'Greens', 'Trees', 'Grass',
                        'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road',
                        'Bright object', 'Water']

    elif method == 'IP':
        HSI = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        GT = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        Num_Classes = 16
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                        'Grass-pasture', 'Grass-trees', 'Grass-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings',
                        'Stone-Steel']
        
    elif method == 'BS':
        HSI = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        GT = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        Num_Classes = 14
        target_names = ['Water', 'Hippo Grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2',
                        'Reeds 1', 'Riparian', 'Firescar 2', 'Island Interior', 'Woodlands',
                        'Acacia Shrublands', 'Acacia Grasslands', 'Short Mopane', 'Mixed Mopane', 'Exposed Soils']

    elif method == 'KSC':
        HSI = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        GT = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        Num_Classes = 13
        target_names = ['Scrub', 'Willow Swamp', 'CP/Oak', 'CP hammock', 'Slash Pine', 'Oak/Broadleaf', 'Hardwood Swamp',
                        'Graminoid Marsh', 'Spartina Marsh', 'Cattail Marsh', 'Salt Marsh', 'Mud Flats', 'Water']
     
    elif method == 'PU':
        HSI = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        GT = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        Num_Classes = 9
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted','Soil','Bitumen',
                        'Bricks','Shadows']

    elif method == 'PC':
        HSI = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        GT = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
        Num_Classes = 9
        target_names = ['Water', 'Trees', 'Asphalt', 'Bricks', 'Bitumen', 'Tiles', 'Shadows',
                        'Meadows', 'Soil']

    elif method == 'SA':
        HSI = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        GT = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        Num_Classes = 16
        target_names = ['Weeds_1','Weeds_2','Fallow',
                        'Fallow_rough_plow','Fallow_smooth', 'Stubble','Celery',
                        'Grapes_untrained','Soil_vinyard_develop','Corn_Weeds',
                        'Lettuce_4wk','Lettuce_5wk','Lettuce_6wk',
                        'Lettuce_7wk', 'Vinyard_untrained','Vinyard_trellis']
      
    elif method == 'SLA':
        HSI = sio.loadmat(os.path.join(data_path, 'SalinasA_corrected.mat'))['salinasA_corrected']
        GT = sio.loadmat(os.path.join(data_path, 'SalinasA_gt.mat'))['salinasA_gt']
        val_old = np.array([0,1,10,11,12,13,14])
        val_new = np.array([0,1,2,3,4,5,6])
        index = np.digitize(GT.ravel(), val_old, right=True)
        GT = val_new[index].reshape(GT.shape)
        Num_Classes = 6
        target_names = ['Brocoli 1', 'Corn weeds', 'Lettuce 4wk', 'Lettuce 5wk',
                       'Lettuce 6wk', 'Lettuce 7wk']

    elif method == 'UH13':
        HSI = sio.loadmat(os.path.join(data_path, 'HU13.mat'))['HSI']
        GT = sio.loadmat(os.path.join(data_path, 'HU13_gt.mat'))['gt']
        Num_Classes = 15
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
                    'Soil', 'Water', 'Residential', 'Commercial', 'Road',
                    'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2',
                    'Tennis Court', 'Running Track']

    elif method == 'UH18':
        HSI = sio.loadmat(os.path.join(data_path, 'HU18.mat'))['img']
        GT = sio.loadmat(os.path.join(data_path, 'HU18_gt.mat'))['gt']
        Num_Classes = 20
        target_names = ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees',
                      'Deciduous trees', 'Bare earth', 'Water', 'Residential buildings',
                      'Non-residential buildings', 'Roads', 'Sidewalks', 'Crosswalks',
                      'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots',
                      'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats']
    else:
        print("Wrong Choice")

    return HSI, GT, Num_Classes, target_names

# %%
## Different Dimensional Reduction Methods
def DLMethod(method, HSI, NC = 75):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    if method == 'PCA': ## PCA
        pca = PCA(n_components = NC, whiten = True)
        RHSI = pca.fit_transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    elif method == 'iPCA': ## Incremental PCA
        n_batches = 256
        inc_pca = IncrementalPCA(n_components = NC)
        for X_batch in np.array_split(RHSI, n_batches):
          inc_pca.partial_fit(X_batch)
        X_ipca = inc_pca.transform(RHSI)
        RHSI = np.reshape(X_ipca, (HSI.shape[0], HSI.shape[1], NC))
    return RHSI

# %%
# %%
def TrTeSplit(HSI, GT, trRatio, vrRatio, teRatio, randomState=345):
    # Split into train and test sets
    Tr, Te, TrC, TeC = train_test_split(HSI, GT, test_size=teRatio,
                                        random_state=randomState, stratify=GT)
    # Calculate the validation ratio based on the updated test and train ratios
    totalTrRatio = trRatio + vrRatio
    new_vrRatio = vrRatio / totalTrRatio
    # Split train set into train and validation sets
    Tr, Va, TrC, VaC = train_test_split(Tr, TrC, test_size=new_vrRatio,
                                        random_state=randomState, stratify=TrC)

    return Tr, Va, Te, TrC, VaC, TeC

# %%
# %%
HSID = "IP" ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
DLM = "PCA" ## "PCA", "iPCA"
WS = 12
teRatio = 0.70
trRatio = 0.50 #0.01 0.50
vrRatio = 0.50 #0.99 0.50
k = 15
# adam = Adam (learning_rate= 0.0001,  weight_decay = 1e-06)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True  # Discrete decay
)
adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# adam = tf.keras.optimizers.legacy.Adam(lr = 0.001, decay = 1e-04)
epochs = 50
batch_size = 56
output_dir = os.path.join(f"SSFK_FKAN_V1_update/{HSID}/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
## Creat Patches for 3D (Spatial-Spectral) Models
def ImageCubes(HSI, GT, WS=WS, removeZeroLabels=True):
    num_rows, num_cols, num_bands = HSI.shape
    margin = int(WS / 2)
    padded_data = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    image_cubes = np.zeros((num_rows * num_cols, WS, WS, num_bands))
    patchesLabels = np.zeros((num_rows * num_cols))
    patchIndex = 0
    for r in range(margin, num_rows + margin):
        for c in range(margin, num_cols + margin):
            cube = padded_data[r - margin: r + margin, c - margin: c + margin, :]
            image_cubes[patchIndex, :, :, :] = cube
            patchesLabels[patchIndex] = GT[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
      image_cubes = image_cubes[patchesLabels>0,:,:,:]
      patchesLabels = patchesLabels[patchesLabels>0]
      patchesLabels -= 1
    return image_cubes, patchesLabels

# %%
## Assigning Class Labels for Final Classification and Confusion Matrices
def ClassificationReports(TeC, Te_Pred, target_names,zero_division=0):
    Te_Pred = np.argmax(Te_Pred, axis=1)
    classification = classification_report(np.argmax(TeC, axis=1), Te_Pred, target_names = target_names,zero_division=zero_division)
    oa = accuracy_score(np.argmax(TeC, axis=1), Te_Pred)
    confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    aa = np.mean(each_acc)
    kappa = cohen_kappa_score(np.argmax(TeC, axis=1), Te_Pred)
    return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100

## Writing Results in CSV files
def CSVResults(file_name, classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa, each_acc):
    classification = str(classification)
    confusion = str(confusion)
    with open(file_name, 'w') as CSV_file:
      CSV_file.write('{} Tr_Time'.format(Tr_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Te_Time'.format(Te_Time))
      CSV_file.write('\n')
      CSV_file.write('{} DL_Time'.format(DL_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Kappa accuracy (%)'.format(kappa))
      CSV_file.write('\n')
      CSV_file.write('{} Overall accuracy (%)'.format(oa))
      CSV_file.write('\n')
      CSV_file.write('{} Average accuracy (%)'.format(aa))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(classification))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(each_acc))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(confusion))
    return CSV_file

## Plot and Save Confusion Matrix
def Conf_Mat(Te_Pred, TeC, target_names):
    plt.rcParams.update({'font.size': 12})
    Te_Pred = np.argmax(Te_Pred, axis=1)
    confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred, labels=np.unique(np.argmax(TeC, axis=1)))
    cm_sum = np.sum(confusion, axis=1, keepdims=True)
    cm_perc = confusion / cm_sum.astype(float) * 100
    annot = np.empty_like(confusion).astype(str)
    nrows, ncols = confusion.shape
    for l in range(nrows):
      for m in range(ncols):
        c = confusion[l, m]
        p = cm_perc[l, m]
        if l == m:
          s = cm_sum[l]
          annot[l, m] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
          annot[l, m] = ''
        else:
          annot[l, m] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(confusion, index=np.unique(target_names), columns=np.unique(target_names))
    return cm, annot
  
## Plot Ground Truths
def GT_Plot(CRDHSI, GT, model, WS, k):
  Predicted = model.predict(CRDHSI)
  Predicted = np.argmax(Predicted, axis=1)
  height, width = np.shape(GT)
  ## Calculate the predicted Ground Truths
  outputs = np.zeros((height, width))
  count = 0
  for AA in range(height):
    for BB in range(width):
      target = int(GT[AA,BB])
      if target == 0:
        continue
      else:
        outputs[AA][BB] = Predicted[count]
        count = count+1
  return outputs


# %%
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
## Reduce the Dimensionality
#if k < HSI.shape[2]:
start = time.time()
RDHSI = DLMethod(DLM, HSI, NC = k)
end = time.time()
DL_Time = end - start
## Create Image Cubes for Model Building
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS = WS)
## Split Train/validation and Test sets
Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)
# Reshape train, validation, and test sets
Tr = Tr.reshape(-1, WS,WS,k,1)  # Flatten input for the model
Va = Va.reshape(-1, WS,WS,k,1)  # Flatten input for the model
Te = Te.reshape(-1, WS,WS,k,1)  # Flatten input for the mode
# Tr = Tr.reshape(-1, WS, WS, k, 1)
TrC = to_categorical(TrC, num_classes=Num_Classes)
# Va = Va.reshape(-1, WS, WS, k, 1)
VaC = to_categorical(VaC, num_classes=Num_Classes)
# Te = Te.reshape(-1, WS, WS, k, 1)
TeC = to_categorical(TeC, num_classes=Num_Classes)
print(Tr.shape)
print(TrC.shape)


# %%
class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.spatial_tokens = Dense(out_channels)
        self.spectral_tokens = Dense(out_channels)
    def call(self, x):
        B, H, W, C = x.shape
        # Use tf.shape to handle dynamic batch dimension
        spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
        spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
        return spatial_tokens,spectral_tokens

# %%
class SplineLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, init_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.weight = self.add_weight(
            shape=(self.out_features, self.in_features),
            initializer=tf.keras.initializers.GlorotUniform(), #tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=init_scale), #HeNormal(),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.001)
        )
    def call(self, inputs):
        return tf.matmul(inputs, tf.transpose(self.weight))
    
class RadialBasisFunction(tf.keras.layers.Layer):
    def __init__(self, grid_min=-2.0, grid_max=2.0, num_grids=15, denominator=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
        self.grid = tf.Variable(
            initial_value=tf.linspace(grid_min, grid_max, num_grids),
            trainable=False,
            name="grid"
        )
    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=-1)
        return tf.exp(-((expanded_inputs - self.grid) / self.denominator) ** 2)
    
class FastKANLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 input_dim, output_dim, 
                 grid_min=-2.0, grid_max=2.0, 
                 num_grids=15, use_base_update=True, 
                 use_layernorm=True, base_activation=tf.nn.relu, 
                 spline_weight_init_scale=0.1, 
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layernorm = use_layernorm
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.layernorm = None
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, init_scale=spline_weight_init_scale)
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            normalized_inputs = self.layernorm(inputs)
        else:
            normalized_inputs = inputs
        spline_basis = self.rbf(normalized_inputs)
        spline_basis_flat = tf.reshape(
            spline_basis,
            shape=(-1, spline_basis.shape[-2] * spline_basis.shape[-1])
        )
        ret = self.spline_linear(spline_basis_flat)
        ret = tf.reshape(ret, shape=tf.shape(inputs)) 
        # ret = tf.reshape(ret, [-1, ret.shape[-1]])  # Adjust as needed, e.g., flattening over axes
        if self.use_base_update:
            base = self.base_linear(self.base_activation(inputs))
            ret = ret + base
        return ret

class FastKAN(tf.keras.Model):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min =-2.0,
        grid_max =2.0,
        num_grids=15,
        use_base_update = True,
        base_activation = tf.nn.silu,
        spline_weight_init_scale = 0.1,
    ):
        super().__init__()
        self.kan_layers = []
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.kan_layers.append(
                FastKANLayer(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
            )

    def call(self, x, training=False):
        for layer in self.kan_layers:
            # residual = x
            x = layer(x, training=training)
            # x = x + residual
        return x


# %%
class MultiHeadKANAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 input_dim, 
                 num_heads, 
                 output_dim, 
                 grid_min=-2.0, 
                 grid_max=2.0, 
                 num_grids=15, 
                 use_base_update=True, 
                 use_layernorm=True, 
                 base_activation=tf.nn.swish, 
                 spline_weight_init_scale=0.1, 
                 dropout_rate=0.1, 
                 output_activation=None,  # Optional activation for output
                 **kwargs):
        super().__init__(**kwargs)
        
        assert output_dim % num_heads == 0, "Output dimension must be divisible by the number of heads."
        
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        
        # Query, Key, and Value projections via FastKANLayer
        self.query_kan = FastKANLayer(
            input_dim=input_dim, 
            output_dim=output_dim, 
            grid_min=grid_min, 
            grid_max=grid_max, 
            num_grids=num_grids, 
            use_base_update=use_base_update, 
            use_layernorm=use_layernorm, 
            base_activation=base_activation, 
            spline_weight_init_scale=spline_weight_init_scale
        )
        self.key_kan = FastKANLayer(
            input_dim=input_dim, 
            output_dim=output_dim, 
            grid_min=grid_min, 
            grid_max=grid_max, 
            num_grids=num_grids, 
            use_base_update=use_base_update, 
            use_layernorm=use_layernorm, 
            base_activation=base_activation, 
            spline_weight_init_scale=spline_weight_init_scale
        )
        self.value_kan = FastKANLayer(
            input_dim=input_dim, 
            output_dim=output_dim, 
            grid_min=grid_min, 
            grid_max=grid_max, 
            num_grids=num_grids, 
            use_base_update=use_base_update, 
            use_layernorm=use_layernorm, 
            base_activation=base_activation, 
            spline_weight_init_scale=spline_weight_init_scale
        )
        
        # Final output projection
        self.output_projection = tf.keras.layers.Dense(
            units=output_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform()  # Better weight initialization
        )
        
        # Optional output activation
        self.output_activation = tf.keras.layers.Activation(output_activation) if output_activation else None

        # Dropout for attention weights
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension of the input into (num_heads, head_dim)
        Transpose the resulting tensor to (batch_size, num_heads, seq_len, head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        """
        Combines the (num_heads, head_dim) into a single dimension.
        Reshapes tensor to (batch_size, seq_len, output_dim).
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads * self.head_dim))

    def call(self, queries, keys, values, training=False):
        batch_size = tf.shape(queries)[0]

        # Compute Query, Key, and Value embeddings using FastKANLayer
        Q = self.query_kan(queries)
        K = self.key_kan(keys)
        V = self.value_kan(values)

        # Split heads into (batch_size, num_heads, seq_len, head_dim)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        attention_scores = tf.matmul(Q, K, transpose_b=True)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        d_k = tf.cast(self.head_dim, tf.float32)  # Head dimension
        attention_scores /= tf.math.sqrt(d_k)  # Scale by sqrt(head_dim) for numerical stability
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # Softmax over sequence length

        # Dropout on attention weights
        attention_weights = self.attention_dropout(attention_weights, training=training)

        # Weighted sum of Value embeddings
        attention_output = tf.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Combine heads back to (batch_size, seq_len, output_dim)
        attention_output = self.combine_heads(attention_output)

        # Final linear projection
        output = self.output_projection(attention_output)

        # Optional activation function
        if self.output_activation:
            output = self.output_activation(output)

        return output



class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, activation="sigmoid", **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        
        # Initialize FastKAN as the excitation mechanism
        self.fastkan = FastKANLayer(
            input_dim=channels,
            output_dim=channels,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=15,
            use_base_update=True,
            base_activation=tf.nn.silu,
            spline_weight_init_scale=0.1  # Smaller initial weight to improve stability
        )
        
        # Final activation (defaults to sigmoid)
        self.activation = tf.keras.layers.Activation(activation)

        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(0.3)  # Dropout can prevent overfitting during training

    def call(self, inputs, training=False):
        x = self.fastkan(inputs)
        x = self.dropout(x, training=training)
        x = self.activation(x)
        return x

class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, reduction_ratio=16, dropout_rate=0.2, **kwargs):
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        
        # Separate SE blocks for spatial and spectral enhancements
        self.spatial_se = SEBlock(out_channels, reduction_ratio)
        self.spectral_se = SEBlock(out_channels, reduction_ratio)
        
        # Layer normalization blocks
        self.norm_spatial = tf.keras.layers.LayerNormalization(axis=-1)
        self.norm_spectral = tf.keras.layers.LayerNormalization(axis=-1)
        
        # Optional Dropout
        self.spatial_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.spectral_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, spatial_tokens, spectral_tokens, center_tokens, training=False):
        # SE weights for spatial and spectral
        spatial_weights = self.spatial_se(center_tokens, training=training)
        spectral_weights = self.spectral_se(center_tokens, training=training)
        
        # Weight application (modulated enhancements)
        spatial_enhanced = spatial_tokens * tf.expand_dims(spatial_weights, 1)
        spectral_enhanced = spectral_tokens * tf.expand_dims(spectral_weights, 1)
        
        # Apply normalization
        spatial_enhanced = self.norm_spatial(spatial_enhanced + spatial_tokens)
        spectral_enhanced = self.norm_spectral(spectral_enhanced + spectral_tokens)
        
        # Optional Dropout (useful for training)
        spatial_enhanced = self.spatial_dropout(spatial_enhanced, training=training)
        spectral_enhanced = self.spectral_dropout(spectral_enhanced, training=training)
    
        return spatial_enhanced, spectral_enhanced

# %%
class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):
        super(StateSpaceModel, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = Dense(units=state_dim, activation="relu")
        self.state_update = Dense(units=state_dim, activation="relu")
    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            # Flatten the input if it has more than 2 dimensions
            input_t = tf.reshape(x[:, t, :], [tf.shape(x)[0], -1])
            state = self.state_transition(state) + self.state_update(input_t)
        return state


# %%
class SSFKSMambaModel(tf.keras.Model):
    def __init__(self, out_channels=64, state_dim=128, num_heads=8, num_grids=15, **kwargs):
        super(SSFKSMambaModel, self).__init__(**kwargs)
        # Tokenization
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        # Multi-Head Attention
        self.multi_head_attention = MultiHeadKANAttention(
            input_dim=out_channels,
            output_dim=out_channels,
            num_heads=num_heads,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=num_grids,
            base_activation=tf.nn.sigmoid
        )
        # Feature Enhancement
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        # State Space Model using Transformer
        self.state_space_model = StateSpaceModel(state_dim=state_dim)
        # Dense Layers and Classification
        self.dense = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')

    def call(self, x, training=False):
        # Step 1: Tokenization
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]  # Center tokens for gating
        # Step 2: Feature Enhancement
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        # Step 3: Apply Multi-Head Attention
        spatial_attention = self.multi_head_attention(spatial_enhanced,spatial_enhanced,spatial_enhanced)
        spectral_attention = self.multi_head_attention(spectral_enhanced,spectral_enhanced,spectral_enhanced)
        # Step 4: Combine the outputs
        combined_output = tf.concat([spatial_attention, spectral_attention], axis=-1)
        # Step 5: State Space Modeling using Transformer
        state_output = self.state_space_model(combined_output)
        # Step 6: Dense layer and classification
        output = self.classifier(state_output)
        return output
def FastKanMamba(Tr, batch_size):
    model = SSFKSMambaModel(
        out_channels=64,
        state_dim=128,
        num_heads=8,
        num_grids=15,
    )
    # Build the model by passing a batch of data
    _ = model(Tr[:batch_size])  # Ensures model is built correctly
    return model


# %%
def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, CRDHSI, HSID, teRatio, k, WS, DLM, RDHSI, GT,batch_size,epochs,output_dir):
    print(f'Model Name : {model_name}')
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)

    if model_name == FastKanMamba:
        Tr = Tr.reshape(-1, WS,WS,k)  # Flatten input for the model
        Va = Va.reshape(-1, WS,WS,k)  # Flatten input for the model
        Te = Te.reshape(-1, WS,WS,k) 
        model = FastKanMamba(Tr, batch_size)
    else:
        model = model_name(WS,k,Num_Classes)
    file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
    # Assuming `model` is your Keras model
    total_params = model.count_params()
    with open(os.path.join(output_dir, file_name),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # Training the model
    start = time.time()
    history = model.fit(x=Tr, y=TrC, batch_size=batch_size, epochs=epochs, validation_data=(Va, VaC))
    end = time.time()
    Tr_Time = end - start
    start = time.time()
    Te_Pre = model.predict(Te)
    end = time.time()
    Te_Time = end - start
    ## Classification Report for Test Model
    classification,Confusion,OA,Per_Class,AA,Kappa = ClassificationReports(TeC, Te_Pre, target_names,zero_division=0)
    Te_Kappa = round(Kappa, 4)
    #Writing Results in CSV File
    file_name = os.path.join(output_dir,f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Classification_Report.csv")
    CSV_file = CSVResults(file_name, classification, Confusion, Tr_Time, Te_Time, DL_Time, Kappa, OA, AA, Per_Class)
    print(classification)
    # files.download(file_name)
    # Ground Truths
    outputs = GT_Plot(CRDHSI, GT, model, WS, k)
    plt.figure(figsize=(10, 10))
    plt.imshow(outputs, cmap='nipy_spectral')
    plt.axis('off')
    file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Ground_Truths.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500,format='png', bbox_inches='tight', pad_inches=0)
    return history,Tr_Time, Te_Time, OA, AA,Kappa,total_params

# %%
# model_names = [FastKanMamba,IN2D,IN3D,HybIN]
model_names = [FastKanMamba] #[CNN2D,IN2D,FastKanMamba] #[FastKanMambaWithAttention,FastKanMamba]
all_results = {}
history_list = []
Tr_Time_list= []
Te_Time_list= []

for model_name in model_names:
  history, Tr_Time, Te_Time, OA, AA,Kappa,total_params = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC,
                                       adam, CRDHSI, HSID, teRatio, k, WS,
                                       DLM, RDHSI, GT, batch_size,epochs,output_dir)
  history_list.append(history)
  Tr_Time_list.append(Tr_Time)
  Tr_Time_list.append(Te_Time)
  
  model_function_name = model_name.__name__
  all_results[model_function_name] = {
        "history": history.history,
        "Tr_Time": Tr_Time,
        "Te_Time": Te_Time,
        "Overall Accuracy": OA,
        "Average Accuracy": AA,
        "Kappa": Kappa,
        "Training Parameters": total_params
    }
  
# Define the filename for all results
all_results_filename = "all_models_results.json"

# Concatenate output_dir with the filename
all_results_output_path = os.path.join(output_dir, all_results_filename)

# Convert the dictionary to JSON format
json_data = json.dumps(all_results, indent=4)

# Optionally, write the JSON data to a file
with open(all_results_output_path, 'w') as json_file:
    json_file.write(json_data)

# %%
# Plot training and validation loss and accuracy for each model
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
colors = ['blue','brown', 'gray', 'green', 'purple', 'orange', 'red']
for i, history in enumerate(history_list):
    # Plot loss
    axs[0].plot(history.history['loss'], label=f'{model_names[i].__name__} Train', color=colors[i])
    axs[0].plot(history.history['val_loss'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
    # Plot accuracy
    axs[1].plot(history.history['accuracy'], label=f'{model_names[i].__name__} Train', color=colors[i])
    axs[1].plot(history.history['val_accuracy'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
axs[0].set_title('Loss', fontsize=10)
axs[0].set_xlabel('Epoch', fontsize=10)
axs[1].set_title('Accuracy', fontsize=10)
axs[1].set_xlabel('Epoch', fontsize=10)
axs[1].legend(fontsize=6)
axs[0].grid(True)
axs[1].grid(True)
plt.tight_layout()
file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_acc_loss_curve_all_models.png"
plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)


