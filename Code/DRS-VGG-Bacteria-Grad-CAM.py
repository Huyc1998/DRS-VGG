#coding=utf-8
from pickle import FALSE
from plistlib import FMT_BINARY
from re import I
from tkinter import Y
from token import LBRACE
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from tables import test
import torch
import itertools
import sklearn
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve, auc,mean_squared_error, r2_score,accuracy_score, average_precision_score,precision_score,f1_score,recall_score, precision_recall_curve
from sklearn.model_selection import  cross_val_score, train_test_split, KFold

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from keras_flops import get_flops
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph 
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation,MaxPooling1D,Flatten,Dropout,LeakyReLU,MaxPool1D
from tensorflow.keras.layers import AveragePooling1D, Input, GlobalAveragePooling1D 
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Lambda


def solve_cudnn_error():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
	  try:
	    # Currently, memory growth needs to be the same across GPUs
	    for gpu in gpus:
	      tf.config.experimental.set_memory_growth(gpu, True)
	    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	  except RuntimeError as e:
	    # Memory growth must be set before GPUs have been initialized
	    print(e)
solve_cudnn_error()
TF_ENABLE_GPU_GARBAGE_COLLECTION=FALSE
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.99)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))  


mpl.rcParams['figure.figsize'] = (10, 8)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):

    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()
        plt.tight_layout()#


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


def topK_accuracy(output, target, topk=(1)):#TOP1top5
    target=lb.inverse_transform(target)
    le = LabelEncoder()
    le.fit(target)
    target=le.transform(target)
    target=torch.from_numpy(target)
    output=torch.from_numpy(output)
    maxk = max(topk)
    batch_size =len(target) #target.size
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))#view(1, -1)
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append("%.6f"%(correct_k / batch_size))
    return res
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
 
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def ALLMETHOD_get_flops(model):
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops 
    return flops
    
# np.set_printoptions(threshold=np.inf)
data_B= np.load('\X_reference.npy',allow_pickle=True)
Lab_B = np.load('\y_reference.npy', allow_pickle=True)
X= np.array(data_B)
y=np.array(Lab_B)

SEED=np.random.randint(10000)
np.random.seed(SEED)
np.random.shuffle(X)
np.random.seed(SEED)
np.random.shuffle(y)
lb = LabelBinarizer()
y = lb.fit_transform(y)
def MaxMinNormalization(x):
     x = (x-np.min(x)) / (np.max(x)-np.min(x))
     return x
np.set_printoptions(threshold=np.inf)

# for i,j in enumerate(X):
#     X[i] = MaxMinNormalization(j)


# define and train a model
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                            downsample_strides=2):

    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        
        # residual = BatchNormalization()(residual)
        # residual = Activation(LeakyReLU(alpha=0.001))(residual)#LeakyReLU(alpha=0.001)
        residual = Conv1D(out_channels, 3, strides=(downsample_strides), 
                        padding='same',activation='relu' ,kernel_initializer='he_uniform',
                        )(residual)
        #kernel_initializer=keras.initializers.he_uniform(seed=None) 'he_uniform'
        # residual = BatchNormalization()(residual)
        # residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, padding='same',activation='relu' ,kernel_initializer='he_uniform')(residual)
        residual = Conv1D(out_channels, 3, padding='same',activation='relu'  ,kernel_initializer='he_uniform')(residual)
        residual = Conv1D(out_channels, 3, padding='same',activation='relu'  ,kernel_initializer='he_uniform')(residual)
        
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling1D()(residual_abs)#GAP#!lemda
        
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, 
                    )(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu' )(scales)
        scales = Dense(out_channels, activation=None, 
                    )(scales)
        scales = Dense(out_channels, activation='sigmoid',)(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])
        
        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = MaxPooling1D(2, strides=2, padding='same')(identity)
            
        # Zero_padding to match channels 
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)    
        residual = keras.layers.add([residual, identity])
    
    return residual
inputs = Input(shape=(1000,1),name='data_input_1')

net = Conv1D(64, 3,strides=1,activation='relu' ,padding='same',name='conv_1')(inputs)
net = Conv1D(64, 3,strides=1,activation='relu', padding='same',)(net)
net = MaxPooling1D(2, strides=2)(net)

net = Conv1D(128, 3, activation='relu',padding='same',)(net)
net = Conv1D(128, 3, activation='relu' ,padding='same',)(net)
net = MaxPooling1D(2, strides=2)(net)


net = residual_shrinkage_block(net, 1, 128, downsample=False)
net = MaxPooling1D(2, strides=2)(net)

net = residual_shrinkage_block(net, 1, 128, downsample=False)
net = MaxPooling1D(2, strides=2)(net)

net = residual_shrinkage_block(net, 1, 128, downsample=False)
net = MaxPooling1D(2, strides=2)(net)

net=Flatten()(net)
net=Dense(1024)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)
# net=Dense(512)(net)#新的
# net = GlobalAveragePooling1D()(net))

net = Dense(1024)(net)#256+512?
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)

outputs = Dense(30, activation='softmax')(net)

model = tf.keras.Model(inputs=inputs,outputs=outputs,name='first_model')
model.summary()

print("The GFLOPs is:{}".format(ALLMETHOD_get_flops(model)/1000000000) ,flush=True )

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
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')
]


train_data, test_data, train_label, test_label = train_test_split(np.array(X), y, train_size = 0.7, test_size = 0.3, stratify=y, shuffle=True)

checkpoint_filepath = "/Bac.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# log_dir = "D:\logs/DRSN/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True,write_images=True,histogram_freq=1)
#tf.keras.utils.plot_model(model, "/model_architecture_diagram11.png", show_shapes=True)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8), metrics=['accuracy'])

history_1 =model.fit(train_data,  train_label, batch_size=512, epochs=40, validation_split=0.2,verbose=1,callbacks=[model_checkpoint_callback])

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
# #plt.savefig('/curve11.jpeg')
# # plot_metrics(history_1)
# plt.show()


# # get results 查看
# K.set_learning_phase(0)
# # DRSN_train_score = model.evaluate(x_train, y_train, batch_size=100, verbose=0)
# # print('Train loss:', DRSN_train_score[0])
# # print('Train accuracy:', DRSN_train_score[1])
# # DRSN_test_score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
# # print('Test loss:', DRSN_test_score[0])
# # print('Test accuracy:', DRSN_test_score[1])
# print(max(history_1.history['val_accuracy']))


model.load_weights("/Bac.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8), metrics=['accuracy'])

results = model.evaluate(test_data,  test_label, batch_size=100, return_dict=True, verbose=2)

test_predictions = model.predict(test_data, batch_size=32)

chineselabel1 = np.zeros([343,343])
np.set_printoptions(threshold = 1e6)
for i in range(343):
    chineselabel1[i,i]=1
chineselabel=lb.inverse_transform(chineselabel1)
print(chineselabel)
input_image=test_data[10,]


def grad_cam(layer_name, data):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    last_conv_layer_output, preds = grad_model(data)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        print(class_channel)
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.reduce_mean(heatmap, axis=(1))
    heatmap = np.expand_dims(heatmap,0)

    heatmap = np.maximum(heatmap,0)
    heatmap/=np.max(heatmap)
    return heatmap
def zero_to_nan(values):#等于左边的值或者等于右边的值
    return [float('nan') if x==0 else x for x in values]


layer_name = "conv1d_14"
#Grad-CAM
for i,j in enumerate(test_data):
    data = np.expand_dims(j,0)

    pred = model.predict(data)[0]
    argmax_num=tf.argmax(pred, 0)
    datax_new=np.load('\wavenumbers.npy', allow_pickle=True)
    pred_label= np.zeros(len(pred))
    pred_label[argmax_num]=1
    #print(f"Model prediction ={pred[argmax_num]}, Predict label = {' '.join(lb.inverse_transform(np.atleast_2d(pred_label)))}, True label = {' '.join(lb.inverse_transform(np.atleast_2d(test_label[i])))}",flush=True)
    
    # pred_2=model.predict(data)[0]
    # pred_2[tf.argmax(pred_2, 0)]=0
    # pred_label2= np.zeros(len(pred_2))
    # pred_label2[tf.argmax(pred_2, 0)]=1
    # print(f"Model prediction ={pred[tf.argmax(pred_2, 0)]}, Predict label2 = {' '.join(lb.inverse_transform(np.atleast_2d(pred_label2)))}, True label = {' '.join(lb.inverse_transform(np.atleast_2d(test_label[i])))}",flush=True)
        
    # pred_3=pred_2
    # pred_3[tf.argmax(pred_3, 0)]=0
    # pred_label3= np.zeros(len(pred_3))
    # pred_label3[tf.argmax(pred_3, 0)]=1
    # print(f"Model prediction ={pred[tf.argmax(pred_3, 0)]}, Predict label3 = {' '.join(lb.inverse_transform(np.atleast_2d(pred_label3)))}, True label = {' '.join(lb.inverse_transform(np.atleast_2d(test_label[i])))}",flush=True)
    if lb.inverse_transform(np.atleast_2d(pred_label))==[14.]:

        plt.figure(figsize=(13,5))
        plt.subplot(1,1,1)

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        plt.tick_params(width=2)
        plt.rc('font',family='Times New Roman')
        plt.xticks(np.linspace(500,1700,6,endpoint=True),fontproperties='Times New Roman', size=35)
        plt.yticks(fontproperties='Times New Roman', size=35)
        plt.yticks([])
        plt.tick_params(labelsize=35)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        heatmap = grad_cam(layer_name,data)
        h=plt.imshow(np.expand_dims(heatmap,axis=2),cmap='Reds', aspect="auto", interpolation='hamming',extent=[381.98,1792.4,j.min(),j.max()], alpha=1)
        plt.plot(datax_new,zero_to_nan(np.squeeze(j)),'k',Linewidth=2)
        plt.title(f"Single Spectrum of MSSA 1",fontdict={'family' : 'Times New Roman', 'size'   : 35},pad=19)
        plt.tight_layout()
        plt.show()


#---------------------------------------
y_true = test_label
y_pred = test_predictions
print('TOPK 1',topK_accuracy(y_pred, y_true,[1]))
print('TOPK 3',topK_accuracy(y_pred, y_true,[3]))
print('TOPK 5',topK_accuracy(y_pred, y_true,[5]))
print('Macro AUC', roc_auc_score(y_true, y_pred))
print('Macro PRC', average_precision_score(y_true, y_pred))

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
print('---------------------------------------')
#print('Classification_report', sklearn_metrics.classification_report(y_true, y_pred, digits=4, zero_division=1))
print('test_accuracy_score', sklearn_metrics.accuracy_score(y_true, y_pred))
print('------Weighted------')
print('Weighted precision', precision_score(y_true, y_pred, average='weighted', zero_division=1))
print('Weighted recall', recall_score(y_true, y_pred, average='weighted', zero_division=1))
print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted', zero_division=1))
