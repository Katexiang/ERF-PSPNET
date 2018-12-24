import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
arg_scope = tf.contrib.framework.arg_scope
def get_conv_arg_scope(is_training, bn=True, reg=None,  use_relu=True, bn_decay=0.9,reuse=None):
    with arg_scope(
        [slim.conv2d,slim.conv2d_transpose],
        padding = "SAME",
        activation_fn = tf.nn.relu if use_relu else None,
        normalizer_fn = slim.batch_norm if bn else None,
        normalizer_params = {"is_training": is_training, "decay": bn_decay},
        weights_regularizer = reg,
        variables_collections = None,
        reuse =reuse
        ) as scope:
        return scope


def downsample(x, n_filters, is_training, bn=False, use_relu=False, l2=None, name="down",reuse=None):
    with tf.variable_scope(name):
        reg = None if l2 is None else slim.l2_regularizer(l2)
        with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training, bn=bn, use_relu=use_relu,reuse=reuse)):
            n_filters_in = x.shape.as_list()[-1]
            n_filters_conv = n_filters - n_filters_in
            x=tf.concat([slim.conv2d(x, n_filters_conv, kernel_size=[3, 3], stride=2,scope='conv'),slim.max_pool2d(x,[2,2],padding='SAME',stride=2,scope='pool')],-1)
    return x
	

	
	
	
def factorized_res_module(x, is_training, dropout=0.3, dilation=[1,1], l2=None, name="fres",reuse=None):
    with tf.variable_scope(name):
        reg = None if l2 is None else slim.l2_regularizer(l2)
        with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training, bn=True,reuse=reuse)):
            n_filters = x.shape.as_list()[-1]
            y = slim.conv2d(x,n_filters,kernel_size=[3,1],rate=dilation[0],normalizer_fn=None,scope='conv_a_3x1')
            y = slim.conv2d(y,n_filters,kernel_size=[1,3],rate=dilation[0],scope='conv_a_1x3')
            y = slim.conv2d(y,n_filters,kernel_size=[3,1],rate=dilation[1],normalizer_fn=None,scope='conv_b_3x1')
            y = slim.conv2d(y,n_filters,kernel_size=[1,3],rate=dilation[1],scope='conv_b_1x3')
            if reuse is  None:
                y = slim.dropout(y,dropout)
            y = tf.add(x,y,name='add')
    return y
   
def Encoder(x, is_training,bn,l2=None,reuse=None):
    #x = tf.div(x, 255., name="rescaled_inputs")
    net=downsample(x, 16, is_training=is_training, bn=bn, use_relu=True, l2=l2, name="d1",reuse=reuse)
    net=downsample(net, 64, is_training=is_training, bn=bn, use_relu=True, l2=l2, name="d2",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 1], l2=l2,name="fres3",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 1], l2=l2,name="fres4",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 1], l2=l2,name="fres5",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 1], l2=l2,name="fres6",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 1],l2=l2, name="fres7",reuse=reuse)
    net=downsample(net, 128, is_training=is_training, bn=bn, use_relu=True, l2=l2, name="d8",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 2],l2=l2, name="fres9",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 4],l2=l2, name="fres10",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 8], l2=l2,name="fres11",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 16], l2=l2,name="fres12",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 2], l2=l2,name="fres13",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 4],l2=l2, name="fres14",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 8],l2=l2, name="fres15",reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dilation=[1, 16],l2=l2, name="fres16",reuse=reuse)
    return net
	
def Decoder(x,numclasses,shape=[480,640],name='decoder',is_training=False,l2=None,bn=True,reuse=None):
    p1=x
    p2=slim.avg_pool2d(x,[2,2],padding='SAME',stride=2,scope='pool2')
    p3=slim.avg_pool2d(x,[4,4],padding='SAME',stride=4,scope='pool3')
    p4=slim.avg_pool2d(x,[8,8],padding='SAME',stride=8,scope='pool4')
    with tf.variable_scope(name):
        reg = None if l2 is None else slim.l2_regularizer(l2)
        with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training,reuse=reuse)):
            j1=slim.conv2d(p1,32,kernel_size=1,scope='conv1')
            j2=slim.conv2d(p2,32,kernel_size=1,scope='conv2')
            j3=slim.conv2d(p3,32,kernel_size=1,scope='conv3')
            j4=slim.conv2d(p4,32,kernel_size=1,scope='conv4')
            f2=tf.image.resize_images(j2, [shape[0]//8,shape[1]//8],method=0)
            f3=tf.image.resize_images(j3, [shape[0]//8,shape[1]//8],method=0)
            f4=tf.image.resize_images(j4, [shape[0]//8,shape[1]//8],method=0)
            net=tf.concat([p1,j1,f2,f3,f4],-1)
            net=slim.conv2d(net,256,kernel_size=3,scope='conv5')
#            if reuse is None:
#                net=slim.dropout(net,0.1)
            net=slim.conv2d(net,numclasses,kernel_size=1,normalizer_fn=None,activation_fn=None,scope='conv6')
            final=tf.image.resize_images(net, [shape[0],shape[1]],method=0)
            probabilities = tf.nn.softmax(final, name='logits_to_softmax')

            
    return final,probabilities
	
def train(x,l2,shape=[480,640],numclasses=66,reuse=None,is_training=True):
    x=Encoder(x, is_training=is_training,bn=True,l2=l2,reuse=reuse)
    x=Decoder(x,numclasses,shape=shape,name='decoder',is_training=is_training,l2=l2,bn=True,reuse=reuse)
    return x