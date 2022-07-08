#coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, BatchNormalization, Activation,Input,GlobalAveragePooling1D 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
import time
import datetime

import torch
import sklearn.metrics as sklearn_metrics

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import shutil
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.core import Lambda
from pickle import FALSE
import tempfile
import tensorflow_model_optimization as tfmot
K.set_learning_phase(1)
def solve_cudnn_error():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
	  try:
	    # Currently, memory growth needs to be the same across GPUs
	    for gpu in gpus:
	      tf.config.experimental.set_memory_growth(gpu, True)
	    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	  except RuntimeError as e:
	    # Memory growth must be set before GPUs have been initialized
	    print(e)
solve_cudnn_error()
TF_ENABLE_GPU_GARBAGE_COLLECTION=FALSE
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.99)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

np.set_printoptions(threshold=np.inf)
mpl.rcParams['figure.figsize'] = (10, 8)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def abs_backend(inputs):
    return abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(inputs,1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs,-1)
    inputs = K.spatial_2d_padding(inputs, ((0,0),(pad_dim,pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)

def topK_accuracy(output, target, topk):
    target=lb.inverse_transform(target)
    le = LabelEncoder()
    le.fit(target)
    target=le.transform(target)
    target=torch.from_numpy(target)
    output=torch.from_numpy(output)
    batch_size =len(target) #target.size
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))#view(1, -1)
    res = []
    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    res.append("%.8f"%(correct_k / batch_size))
    return res

def ALLMETHOD_get_flops(model):
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops 
    return flops

def del_file(filepath):
  
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  
def get_gzipped_model_size(file):
# Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)
data_5618= np.load('D:\XXXX\XXXX/XXXX/Data56181601.npy',allow_pickle=True)
Lab_5618 = np.load('D:\XXXX\XXXX/XXXX/Lab56181601.npy', allow_pickle=True)
X= np.array(data_5618)
y=np.array(Lab_5618)
SEED=np.random.randint(100000)
np.random.seed(SEED)
np.random.shuffle(X)
np.random.seed(SEED)
np.random.shuffle(y)
lb = LabelBinarizer()
y = lb.fit_transform(y)
train_data, test_data, train_label, test_label = train_test_split(np.array(X), y, train_size = 0.7, test_size = 0.3, stratify=y, shuffle=True)
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, train_size = 0.8, test_size = 0.2, stratify=train_label, shuffle=True)
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.CategoricalAccuracy(name='accuracy'),
    keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
    keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
    keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),

]

T1_list = []
T3_list = []
T5_list = []
PRE_list = []
RECALL_list = []
F1_list = []
time_list = []
BA= []
PA= []
d={}


def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                            downsample_strides=2):

    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        

        residual = Conv1D(out_channels, 3, strides=(downsample_strides), 
                        padding='same',activation='relu' ,kernel_initializer='he_uniform',
                        )(residual)
        residual = Conv1D(out_channels, 3, padding='same',activation='relu' ,kernel_initializer='he_uniform')(residual)
        residual = Conv1D(out_channels, 3, padding='same',activation='relu'  ,kernel_initializer='he_uniform')(residual)
        residual = Conv1D(out_channels, 3, padding='same',activation='relu'  ,kernel_initializer='he_uniform')(residual)
        
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling1D()(residual_abs)
        
        scales = Dense(out_channels, activation=None, 
                    )(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu' )(scales)
        scales = Dense(out_channels, activation=None, 
                    )(scales)
        scales = Dense(out_channels, activation='sigmoid',)(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        thres = keras.layers.multiply([abs_mean, scales])
        
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        if downsample_strides > 1:
            identity = MaxPooling1D(2, strides=2, padding='same')(identity)
            
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)    
        residual = keras.layers.add([residual, identity])
    
    return residual
inputs = Input(shape=(1601,1),name='data_input_1')
net = Conv1D(64, 3,strides=1,activation='relu' ,padding='same')(inputs)
net = Conv1D(64, 3,strides=1,activation='relu', padding='same',)(net)
net = MaxPooling1D(2, strides=2)(net)

net = Conv1D(128, 3, activation='relu',padding='same',)(net)
net = Conv1D(128, 3, activation='relu' ,padding='same',)(net)
net = MaxPooling1D(2, strides=2)(net)


net = residual_shrinkage_block(net, 1, 128, downsample=False)
net = MaxPooling1D(2, strides=2)(net)#duode在

net = residual_shrinkage_block(net, 1, 128, downsample=False)
net = MaxPooling1D(2, strides=2)(net)

net = residual_shrinkage_block(net, 1, 128, downsample=False)
net = MaxPooling1D(2, strides=2)(net)

net=Flatten()(net)
net=Dense(1024)(net)
net = BatchNormalization()(net)
net = Activation('relu' )(net)
net = Dropout(0.5)(net)

net = Dense(1024)(net)
net = BatchNormalization()(net)
net = Activation('relu' )(net)
net = Dropout(0.5)(net)

outputs = Dense(343, activation='softmax')(net)

model = tf.keras.Model(inputs=inputs,outputs=outputs,name='first_model')
model.summary()

print("The GFLOPs is:{}".format(ALLMETHOD_get_flops(model)/1000000000) ,flush=True )

checkpoint_filepath = "D:\\XXXX\\XXX\\DRS-VGG.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=2, patience=25, mode='max',min_delta=0.0001, restore_best_weights=True)

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
log_dir = "D:\logs/DRS-VGG/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True,write_images=True,histogram_freq=1)
# tf.keras.utils.plot_model(model, "D:\XXXX\XXXXX/model_architecture_diagram.png", show_shapes=True)
t=time.time()
history_1 = model.fit(train_data, train_label, batch_size=64, epochs=150, verbose=2, validation_data=(val_data, val_label ),callbacks=[model_checkpoint_callback,early_stopping,tensorboard_callback])
T=time.time()-t

# acc = history_1.history['accuracy']
# val_acc = history_1.history['val_accuracy']
# loss = history_1.history['loss']
# val_loss = history_1.history['val_loss']

# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2) 
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# #plt.savefig('D:\XXXX/XXXX/curve.jpeg')
# plt.show()

#Evaluation
model.load_weights("D:\\XXXX\\XXX\\DRS-VGG.hdf5")
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics= ['accuracy'],
)
test_predictions = model.predict(test_data, batch_size=32)
y_true = test_label
y_pred = test_predictions
#print(topK_accuracy(y_pred, y_true,1))
T1_list.append(topK_accuracy(y_pred, y_true,1))      
T3_list.append(topK_accuracy(y_pred, y_true,3))
T5_list.append(topK_accuracy(y_pred, y_true,5))

for i in range(len(y_pred)):
    max_value=max(y_pred[i])
    if max_value == 0:
        print(i, ":")
        print("y_true:")
        print(y_true[i])
        print("y_pred:")
        print(y_pred[i])

    for j in range(len(y_pred[i])):
        if max_value==y_pred[i][j]:
            y_pred[i][j]=1
        else:
            y_pred[i][j]=0

PRE_list.append(sklearn_metrics.precision_score(y_true, y_pred, labels=None, average='weighted', sample_weight=None, zero_division=1))
RECALL_list.append(sklearn_metrics.recall_score(y_true, y_pred, labels=None, average='weighted', sample_weight=None, zero_division=1))
F1_list.append(sklearn_metrics.f1_score(y_true, y_pred, labels=None,  average='weighted', sample_weight=None, zero_division=1))
time_list.append(T)
T1_list=np.array(T1_list).astype(np.float64)
T3_list=np.array(T3_list).astype(np.float64)
T5_list=np.array(T5_list).astype(np.float64)
PRE_list=np.array(PRE_list).astype(np.float64)
RECALL_list=np.array(RECALL_list).astype(np.float64)
F1_list=np.array(F1_list).astype(np.float64)
time_list=np.array(time_list).astype(np.float64)
print('T1 mean',np.mean(T1_list ),'standard deviation',np.std(T1_list))
print('T3 mean',np.mean(T3_list),'standard deviation',np.std(T3_list))
print('T5 mean',np.mean(T5_list),'standard deviation',np.std(T5_list))
print('Pre mean',np.mean(PRE_list),'standard deviation',np.std(PRE_list))
print('Recall mean',np.mean(RECALL_list),'standard deviation',np.std(RECALL_list))
print('F1 mean',np.mean(F1_list),'standard deviation',np.std(F1_list))
print('TIME mean',np.mean(time_list),'standard deviation',np.std(time_list))


#Weight Pruning
raw_file = 'D:\XXXX\h5/DRS-VGGraw.h5'
tf.keras.models.save_model(model, raw_file, include_optimizer=False)
batch_size=64
epochs=15
train_num=train_data.shape[0]
end_step = np.ceil(train_num / batch_size).astype(np.int32) * epochs

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
initial_sparsity=0, final_sparsity=0.30,
begin_step=0, end_step=end_step)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model,pruning_schedule=pruning_schedule)
model_for_pruning.compile(
    optimizer=keras.optimizers.Adam(lr=0.0008),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics= ['accuracy'],
)
#model_for_pruning.summary()

log_dir2 = "D:\logs/DRS-VGG2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, write_grads=True,write_images=True,histogram_freq=1)

callbacks = [
tfmot.sparsity.keras.UpdatePruningStep(),]
model_for_pruning.fit(train_data,  train_label, batch_size=batch_size, epochs=epochs+5, verbose=0,validation_data=(val_data, val_label), callbacks=[callbacks,tensorboard_callback2])


final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
final_model.compile(
    optimizer=keras.optimizers.Adam(lr=0.0008),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics= ['accuracy'],
)

_,model_for_pruning_accuracy = final_model.evaluate(test_data, test_label, verbose=0)
print('Pruned test accuracy:', model_for_pruning_accuracy)

#Checkout
# for i, w in enumerate(final_model.get_weights()):
#     print(
#         "{} -- Total:{}, Zeros: {:.2f}%".format(
#             final_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
#         )
#     )

#final_model.summary()

pruning_file = 'D:\XXXX\h5/DRS-VGGpru.h5'
tf.keras.models.save_model(final_model, pruning_file, include_optimizer=False)
#print('Saved pruning model to:', pruning_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(raw_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruning_file)))


