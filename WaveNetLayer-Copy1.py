from __future__ import division
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import cPickle
import numpy as np
import theano.tensor as T

rng = np.random.RandomState(1234)

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
    
    
    
    