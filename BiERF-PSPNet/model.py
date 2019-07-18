import tensorflow as tf
def conv(inputs,filters,kernel_size,strides=(1, 1),padding='SAME',dilation_rate=(1, 1),activation=tf.nn.relu,use_bias=True,regularizer=None,name=None,reuse=None):
    out=tf.layers.conv2d(
    inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    dilation_rate=dilation_rate,
    activation=activation,
    use_bias=use_bias,
    kernel_regularizer=regularizer,
    bias_initializer=tf.zeros_initializer(),
    kernel_initializer= tf.random_normal_initializer(stddev=0.1),
    name=name,
    reuse=reuse)
    return out

def batch(inputs,training=True,reuse=None,momentum=0.9,name='n'):
    out=tf.layers.batch_normalization(inputs,training=training,reuse=reuse,momentum=momentum,name=name)
    return out

def downsample(x, n_filters, is_training,l2=None, name="down",momentum=0.9,reuse=None):
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        n_filters_in = x.shape.as_list()[-1]
        n_filters_conv = n_filters - n_filters_in
        x=tf.concat([conv(x, n_filters_conv, kernel_size=[3, 3],activation=None,strides=2,name='conv',regularizer=reg,reuse=reuse),tf.layers.max_pooling2d(x,[2,2],padding='SAME',strides=2,name='pool')],-1)
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='batch'))                     
    return x
	
def factorized_res_module(x, is_training, dropout=0.3, dilation=[1,1], l2=None, name="fres",reuse=None,momentum=0.9):
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        n_filters = x.shape.as_list()[-1]
        y =conv(x,n_filters,kernel_size=[3,1],dilation_rate=dilation[0],name='conv_a_3x1',regularizer=reg,reuse=reuse)    
        y =conv(y,n_filters,kernel_size=[1,3],dilation_rate=dilation[0],activation=None,name='conv_a_1x3',regularizer=reg,reuse=reuse)
        y=tf.nn.relu(batch(y,training=is_training,reuse=reuse,momentum=momentum,name='batch1'))
        y = conv(y,n_filters,kernel_size=[3,1],dilation_rate=[dilation[1],1],name='conv_b_3x1',regularizer=reg,reuse=reuse)
        y = conv(y,n_filters,kernel_size=[1,3],dilation_rate=[1,dilation[1]],activation=None,name='conv_b_1x3',regularizer=reg,reuse=reuse)
        y=tf.nn.relu(batch(y,training=is_training,reuse=reuse,momentum=momentum,name='batch2'))
        y=tf.layers.dropout(y,rate=dropout,training=is_training)
        y =tf.add(x,y,name='add')
    return y

def Encoder(x, is_training,l2=None,reuse=None,momentum=0.9):
    #x = tf.div(x, 255., name="rescaled_inputs")
    net=downsample(x, 16, is_training=is_training,name="d1",momentum=momentum,reuse=reuse)
    net=downsample(net, 64, is_training=is_training,name="d2",momentum=momentum,reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dropout=0.03, dilation=[1, 1], l2=l2,name="fres3",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.03, dilation=[1, 1], l2=l2,name="fres4",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.03, dilation=[1, 1], l2=l2,name="fres5",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.03, dilation=[1, 1], l2=l2,name="fres6",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.03, dilation=[1, 1], l2=l2,name="fres7",reuse=reuse,momentum=momentum)
    net=downsample(net, 128, is_training=is_training,name="d8",momentum=momentum,reuse=reuse)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 2], l2=l2,name="fres9",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 4], l2=l2,name="fres10",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 8], l2=l2,name="fres11",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 16], l2=l2,name="fres12",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 2], l2=l2,name="fres13",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 4], l2=l2,name="fres14",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 8], l2=l2,name="fres15",reuse=reuse,momentum=momentum)
    net = factorized_res_module(net, is_training=is_training, dropout=0.3, dilation=[1, 16], l2=l2,name="fres16",reuse=reuse,momentum=momentum)
    return net
	
def spatial(x,name='spatial',is_training=False,l2=None,reuse=None,momentum=0.9):
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        x = conv(x,64,kernel_size=3,name='conv1',activation=None,regularizer=reg,reuse=reuse,strides=2)
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='batch1'))
        x = conv(x,128,kernel_size=3,name='conv2',activation=None,regularizer=reg,reuse=reuse,strides=2)
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='batch2'))
        x = conv(x,256,kernel_size=3,name='conv3',activation=None,regularizer=reg,reuse=reuse,strides=2)
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='batch3'))
    return x

def Decoder(x,shape=[480,640],name='decoder',is_training=False,l2=None,reuse=None,momentum=0.9):
    p1=x
    p2=tf.layers.average_pooling2d(x,pool_size=[2,2],strides=2,padding='SAME',name='pool2')
    p3=tf.layers.average_pooling2d(x,pool_size=[4,4],strides=4,padding='SAME',name='pool3')
    p4=tf.layers.average_pooling2d(x,pool_size=[8,8],strides=8,padding='SAME',name='pool4')
    with tf.variable_scope(name):
        reg = None if l2 is None else  tf.contrib.layers.l2_regularizer(scale=l2)
        j1=conv(p1, 32, kernel_size=1,activation=None,name='conv1',regularizer=reg,reuse=reuse,use_bias=None)
        j1=tf.nn.relu(batch(j1,training=is_training,reuse=reuse,momentum=momentum,name='batch1'))
        j2=conv(p2, 32, kernel_size=1,activation=None,name='conv2',regularizer=reg,reuse=reuse,use_bias=None)
        j2=tf.nn.relu(batch(j2,training=is_training,reuse=reuse,momentum=momentum,name='batch2'))
        j3=conv(p3, 32, kernel_size=1,activation=None,name='conv3',regularizer=reg,reuse=reuse,use_bias=None)
        j3=tf.nn.relu(batch(j3,training=is_training,reuse=reuse,momentum=momentum,name='batch3'))        
        j4=conv(p4, 32, kernel_size=1,activation=None,name='conv4',regularizer=reg,reuse=reuse,use_bias=None)
        j4=tf.nn.relu(batch(j4,training=is_training,reuse=reuse,momentum=momentum,name='batch4'))           
        f2=tf.image.resize_images(j2, [shape[0]//8,shape[1]//8],method=0)
        f3=tf.image.resize_images(j3, [shape[0]//8,shape[1]//8],method=0)
        f4=tf.image.resize_images(j4, [shape[0]//8,shape[1]//8],method=0)
        net=tf.concat([p1,j1,f2,f3,f4],-1)
        net=conv(net, 256, kernel_size=3,activation=None,name='conv5',regularizer=reg,reuse=reuse,use_bias=None)
        net=tf.nn.relu(batch(net,training=is_training,reuse=reuse,momentum=momentum,name='batch5'))         
    return net
	
def FeatureFusionModule(input_1, input_2, numclasses,name='fusion',is_training=False,l2=None,reuse=None,shape=[360,720],momentum=0.9):
    inputs = tf.concat([input_1, input_2], axis=-1)
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        inputs = conv(inputs, 256, kernel_size=3,activation=None,name='conv1',regularizer=reg,reuse=reuse)
        inputs=tf.nn.relu(batch(inputs,training=is_training,reuse=reuse,momentum=momentum,name='batch1'))
        inputs = conv(inputs, numclasses, kernel_size=3,activation=None,name='conv2',regularizer=reg,reuse=reuse)
        inputs=tf.nn.relu(batch(inputs,training=is_training,reuse=reuse,momentum=momentum,name='batch2'))
        # Global average pooling
        net = tf.reduce_mean(inputs, [1, 2], keepdims=True)
        net = conv(net, numclasses, kernel_size=1,activation=None,name='conv3',regularizer=reg,reuse=reuse)
        net = tf.nn.relu(batch(net,training=is_training,reuse=reuse,momentum=momentum,name='batch3'))
        net = conv(net, numclasses, kernel_size=1,activation=None,name='conv4',regularizer=reg,reuse=reuse)
        net = tf.sigmoid(net)
        net = tf.multiply(inputs, net)
        net = tf.add(inputs, net)
        net = conv(net, numclasses, kernel_size=1,activation=None,name='conv5',regularizer=reg,reuse=reuse)#最后加的
        net = tf.nn.relu(batch(net,training=is_training,reuse=reuse,momentum=momentum,name='batch4'))#最后加的
        net = tf.image.resize_images(net, [shape[0],shape[1]],method=0)
        #net = conv(net, numclasses, kernel_size=1,activation=None,name='conv5',regularizer=reg,reuse=reuse)
    return net

def erfpspcontext(x1,x2,l2,shape=[480,640],shape2=[1024,2048],numclasses=66,reuse=None,is_training=True,momentum=0.9):
    x1 = Encoder(x1,is_training=is_training,l2=l2,reuse=reuse,momentum=momentum)
    x1 = Decoder(x1,shape=shape,is_training=is_training,l2=l2,reuse=reuse,momentum=momentum)
    x1 = tf.image.resize_images(x1, [shape2[0]//8,shape2[1]//8],method=0)
    x2 = spatial(x2,is_training=is_training,l2=l2,reuse=reuse,momentum=momentum)
    y =FeatureFusionModule(x1, x2, numclasses,is_training=is_training,l2=l2,reuse=reuse,shape=shape,momentum=momentum)
    probabilities = tf.nn.softmax(y, name='logits_to_softmax')
    return y,probabilities

	
	
