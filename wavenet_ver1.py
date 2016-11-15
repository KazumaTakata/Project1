from __future__ import division
# coding: utf-8

# In[1]:

import os, sys
#sys.path.append(os.getcwd())

import numpy
numpy.random.seed(123)
import random
random.seed(123)

import theano
import theano.tensor as T
import lib
import lasagne
import scipy.misc

import time
import functools
import itertools

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(seed=4884)
#get_ipython().magic(u'matplotlib inline')


from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import cPickle
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

rng = np.random.RandomState(1234)


# In[5]:

x  = T.tensor4('x')
#x.tag.test_value = train_data.reshape(20,1,96000,1)
#x.tag.test_value = np.random.rand(1, 3, 96000, 1).astype('float32')


# In[6]:

#theano.config.compute_test_value = 'warn'


# In[7]:

class DilatedConv1D:
    def __init__(self, output_dim, input_dim, filter_size, dilation, activation = lambda x : x, mask_type=None, bias_apply = False):
        # accutual filter_size is filter_size // 2 + 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        self.dilation = dilation
        self.mask_type = mask_type
        self.filter_shape = (output_dim, input_dim, filter_size, 1)
        self.bias_apply = bias_apply
        
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]))
        
        self.W = theano.shared(rng.uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)), size=self.filter_shape).astype("float32"), name="W")
        
        if bias_apply is not False:
            self.b = theano.shared(np.zeros(output_dim, dtype=theano.config.floatX), name = 'b')
        
        self.activation = activation
        
        if mask_type is not None:
            mask = np.ones((output_dim, input_dim, filter_size, 1), dtype=theano.config.floatX)
            center = filter_size//2
            for i in xrange(filter_size):
                if (i > center):
                    mask[:, :, i, :] = 0.
            self.W_mask = self.W * mask
            
        if bias_apply is not False:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
            
    def f_prop(self, x):
        # inputs.shape: (batch size, input_channel, length, 1)
        if self.mask_type is not None:
            result = T.nnet.conv2d(x, self.W_mask, border_mode='half', filter_flip=False, filter_dilation=(self.dilation, 1))
        else:
            result = T.nnet.conv2d(x, self.W, border_mode='half', filter_flip=False, filter_dilation=(self.dilation, 1))
        if self.bias_apply is not False:
            result = result + self.b[np.newaxis, :, np.newaxis, np.newaxis] 
        result = self.activation(result)
        return result


# In[87]:

#x.tag.test_value = train_data.reshape(1,1,96000,1)


# In[283]:

#x  = T.tensor4('x')
#x.tag.test_value = np.random.rand(1, 1, 96000, 1).astype('float32')


# In[8]:

class residual_block:
    def __init__(self, DIM, dilation):
        self.tanh_out = DilatedConv1D(DIM, DIM, 5, dilation, activation = T.tanh, mask_type = 'a')
        self.sig_out = DilatedConv1D(DIM, DIM, 5, dilation, activation = T.nnet.sigmoid, mask_type = 'a')
        filter_shape = (DIM, DIM, 1, 1)
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        self.conv1_1_W_skip = theano.shared(rng.uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)), size=filter_shape).astype("float32"), name="W_skip")
        self.conv1_1_W_out = theano.shared(rng.uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)), size=filter_shape).astype("float32"), name="W_out")
        self.params = self.tanh_out.params + self.sig_out.params + [self.conv1_1_W_skip, self.conv1_1_W_out]
    def f_prop(self, x):
        z = self.tanh_out.f_prop(x) * self.sig_out.f_prop(x)
        skip_out = T.nnet.conv2d(z , self.conv1_1_W_skip)
        out = T.nnet.conv2d(z, self.conv1_1_W_out)
        out = out + x 
        return out, skip_out


# In[9]:

num_stack = 9


# In[10]:

model = []
for i in range(num_stack):
    model += [residual_block(256, 2**i)]    


# In[11]:

causal_layer = DilatedConv1D(256, 1, 5, 1, activation = lambda x:x, mask_type = 'a')


# In[12]:

out = causal_layer.f_prop(x)


# In[ ]:

params = []
skip_out = 0
out = out
for layer in model: 
    params += layer.params
    out, skip = layer.f_prop(out)
    skip_out += skip

params = params[:-1]
params += causal_layer.params


# In[14]:

#out.tag.test_value.shape


# In[290]:

#params


# In[291]:

class output_layer:
    def __init__(self):
        filter_shape = (256, 256, 1, 1)
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        self.W_con1 = theano.shared(rng.uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)), size=filter_shape).astype("float32"), name="W_1")
        self.W_con2 = theano.shared(rng.uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)), size=filter_shape).astype("float32"), name="W_2")
        self.params = [self.W_con1, self.W_con2]
        
    def f_prop(self, x):
        return T.nnet.conv2d(T.nnet.relu(T.nnet.conv2d(T.nnet.relu(x) , self.W_con1)), self.W_con2)


# In[293]:

output_Layer = output_layer()
result = output_Layer.f_prop(skip_out)
params += output_Layer.params


# In[274]:

#result.tag.test_value.shape


# In[295]:

tmp_1 = result.reshape((result.shape[0], result.shape[1], result.shape[2]))
tmp_2 = tmp_1.dimshuffle(0,2,1)
y = tmp_2.reshape((-1, tmp_2.shape[2]))
y = T.nnet.softmax(y)
#y.tag.test_value


# In[296]:

raw_inputs = T.vector('raw_inputs').astype('int64')
#tt= np.zeros(1*96000).astype('int64')
#raw_inputs.tag.test_value = tt


# In[199]:

#raw_inputs.tag.test_value.shape


# In[212]:

#cost.tag.test_value


# In[297]:

cost = T.mean(T.nnet.categorical_crossentropy(y, raw_inputs))


# In[311]:

def sgd(cost, params, eps=np.float32(0.1)):
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        updates[param] = param - eps*gparam
    return updates


# In[298]:

def TNorm(x):
    return T.sqrt(T.sum(T.sqr(x)))

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(np.float32(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            norm = TNorm(g)
            g = T.switch(T.lt(1,norm), g/norm, g)
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates


# In[313]:




# In[312]:

#g_params = T.grad(cost, params)
updates = Adam(cost, params)
train = theano.function(inputs=[x, raw_inputs], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
#valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
#test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')

raw_data, train_data = lib.wav.generate_input_data()
print raw_data.shape
print train_data.shape
# In[308]:

TRAIN = train_data.reshape(20,1,96000,1)
tt= raw_data.reshape(20*96000)

BATCH = 2
TRAIN = TRAIN[:BATCH,:,:,:]


cost = 0
for i in range(10000):
    cost = train(TRAIN[:,:,:,:], tt[:BATCH*96000])
    print cost


# In[276]:

#tt= raw_data.reshape(20*96000)


# In[240]:

#TRAIN = train_data.reshape(20,1,96000,1)


# In[306]:

#tt[:20*2000].shape


# In[309]:

#train_data


# In[304]:

#gg = TRAIN[:,:,:2000,:].


# In[292]:

#skip_out.tag.test_value.shape


# In[4]:

raw_data, train_data = lib.wav.generate_input_data()
print raw_data.shape
print train_data.shape


# In[280]:

#tt.min()


# In[301]:

#raw_data.shape


# In[ ]:



