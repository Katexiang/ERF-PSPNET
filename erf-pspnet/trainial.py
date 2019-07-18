import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import model
import random
import csv
import math

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/camvid', 'The log directory to save your checkpoint and event files.')
#Training arguments
flags.DEFINE_integer('num_classes', 11, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 8, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 24, 'The batch size used for validation.')
flags.DEFINE_integer('image_height',360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 1e-3, 'The initial learning rate for your training.')
flags.DEFINE_integer('Start_train',True, "The input height of the images.")

FLAGS = flags.FLAGS

Start_train = flags.Start_train
log_name = 'model.ckpt'

num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
eval_batch_size = FLAGS.eval_batch_size 
image_height = FLAGS.image_height
image_width = FLAGS.image_width

#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8


#Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, 'train', file) for file in os.listdir(dataset_dir + "/train") if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, "trainannot", file) for file in os.listdir(dataset_dir + "/trainannot") if file.endswith('.png')])
image_val_files = sorted([os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "/val") if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "/valannot") if file.endswith('.png')])
#保存到excel
csvname=logdir[6:]+'.csv'
with  open(csvname,'a', newline='') as out:
    csv_write = csv.writer(out,dialect='excel')
    a=[str(i) for i in range(num_classes)]
    csv_write.writerow(a)
#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = math.ceil(len(image_files) / batch_size)
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#=================CLASS WEIGHTS===============================
#Median frequency balancing class_weights

class_weights1=np.array([   4.57716287, 0,  0, 0,  9.60914246, 0, 0, 0 , 0,0,6.10711717], dtype=np.float32)
class_weights2=np.array([  0, 42.28705255,  3.46893819, 16.45916311,  0,0, 33.06296333, 0 , 0,0,0], dtype=np.float32) 
class_weights3=np.array([   0, 0,  0, 0,  0, 33.93236668, 0, 13.5811212 , 40.96211531,44.98280801,0], dtype=np.float32) 

def weighted_cross_entropy(onehot_labels, logits, class_weights):
    #a=tf.reduce_sum(-tf.log(tf.clip_by_value(logits, 1e-10, 1.0))*(1-logits)*(1-logits)*onehot_labels*class_weights)
    a=tf.reduce_sum(-tf.log(tf.clip_by_value(logits, 1e-10, 1.0))*onehot_labels*class_weights)
    return a


def decode(a,b):
    a = tf.read_file(a)
    a=tf.image.decode_png(a, channels=3)
    a = tf.image.convert_image_dtype(a, dtype=tf.float32)
    b = tf.read_file(b)
    b = tf.image.decode_png(b,channels=1)
    bb,bs,_=tf.image.sample_distorted_bounding_box(tf.shape(b),bounding_boxes=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4]),min_object_covered=1.0)
    a=tf.slice(a,bb,bs)
    b=tf.slice(b,bb,bs)
    a=tf.image.resize_images(a, [image_height,image_width],method=0)
    b=tf.image.resize_images(b, [image_height,image_width],method=1)     
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width,1))
    return a,b
def decodev(a,b):
    a = tf.read_file(a)
    a=tf.image.decode_png(a, channels=3)
    a = tf.image.convert_image_dtype(a, dtype=tf.float32)
    b = tf.read_file(b)
    b = tf.image.decode_png(b,channels=1)
    a=tf.image.resize_images(a, [image_height,image_width],method=0)
    b=tf.image.resize_images(b, [image_height,image_width],method=1)     
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width,1))
    return a,b
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TRAINING BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        tdataset = tf.data.Dataset.from_tensor_slices((images,annotations))
        tdataset = tdataset.map(decode)
        tdataset = tdataset.shuffle(100).batch(batch_size).repeat(num_epochs)
        titerator = tdataset.make_initializable_iterator()
        images,annotations = titerator.get_next()		

		
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        vdataset = tf.data.Dataset.from_tensor_slices((images_val,annotations_val))
        vdataset = vdataset.map(decodev)
        vdataset = vdataset.batch(eval_batch_size).repeat(num_epochs*3)
        viterator = vdataset.make_initializable_iterator()
        images_val,annotations_val = viterator.get_next()				
		
		
		
		
		
        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        _, probabilities= model.erfpsp(images,numclasses=num_classes, shape=[image_height,image_width], l2=weight_decay,reuse=None,is_training=True)
        annotations = tf.reshape(annotations, shape=[-1, image_height, image_width])
        raw_gt = tf.reshape(annotations, [-1,])
        indices = tf.squeeze(tf.where(tf.greater(raw_gt,0)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)	
        gt_one = tf.one_hot(gt, num_classes, axis=-1)		
        raw_prediction = tf.reshape(probabilities, [-1, num_classes])
        prediction = tf.gather(raw_prediction, indices)
    
	
		
		
        annotations_ohe = tf.one_hot(annotations, num_classes+1, axis=-1)
        MASK = tf.reduce_sum(1-annotations_ohe[:,:,:,0])	
        m=tf.split(annotations_ohe,num_or_size_splits=num_classes+1,axis=-1)
        M1=tf.reduce_sum(tf.concat([m[2],m[3],m[4],m[6],m[7],m[8],m[9],m[10]],axis=-1),-1,keepdims=True)
        M2=tf.reduce_sum(tf.concat([m[6],m[8],m[9],m[10]],axis=-1),-1,keepdims=True)
        X_=tf.reduce_sum(probabilities*annotations_ohe[:,:,:,1:],-1,keepdims=True)
        mask=tf.reshape(1-annotations_ohe[:,:,:,0],shape=[-1,image_height,image_width,1])
        f1=tf.reduce_sum(tf.pow(tf.sqrt(M1+0.5)*(X_-M1)*mask,2))/(2*MASK)
        f2=tf.reduce_sum(tf.pow(tf.sqrt(M2+0.5)*(X_-M2)*M1,2))/(2*tf.reduce_sum(M1))	

        loss1=weighted_cross_entropy(gt_one, prediction, class_weights1)
        loss2=weighted_cross_entropy(gt_one, prediction, class_weights2)
        loss3=weighted_cross_entropy(gt_one, prediction, class_weights3)	
        los=(loss1+loss2*(f1+2)+loss3*(f1+2)*(f2+2))/YANMO		
        loss=tf.losses.add_loss(los)
        total_loss = tf.losses.get_total_loss()
        global_step =  tf.train.get_or_create_global_step()
        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates_op = tf.group(*update_ops)
        #Create the train_op.
        with tf.control_dependencies([updates_op]):
            train_op = optimizer.minimize(total_loss,global_step=global_step)
        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        #这一块为验证
        _, probabilities_val= model.erfpsp(images_val,numclasses=num_classes, shape=[image_height,image_width], l2=None,reuse=True,is_training=None)
        raw_gt_v = tf.reshape(tf.reshape(annotations_val, shape=[-1, 1024, 2048]),[-1,])
        indices_v = tf.squeeze(tf.where(tf.greater(raw_gt_v,0)), 1)
        gt_v = tf.cast(tf.gather(raw_gt_v, indices_v), tf.int32)
        gt_v = gt_v-1
        gt_one_v = tf.one_hot(gt_v, num_classes, axis=-1)
        raw_prediction_v = tf.argmax(tf.reshape(probabilities_val, [-1, num_classes]),-1)
        prediction_v = tf.gather(raw_prediction_v, indices_v)
        prediction_ohe_v = tf.one_hot(prediction_v, num_classes, axis=-1)
        and_val=gt_one_v*prediction_ohe_v
        and_sum=tf.reduce_sum(and_val,[0])
        or_val=tf.to_int32((gt_one_v+prediction_ohe_v)>0.5)
        or_sum=tf.reduce_sum(apor,axis=[0])
        T_sum=tf.reduce_sum(gta_v,axis=[0])
        R_sum = tf.reduce_sum(prediction_ohe_v,axis=[0])		
        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step ,loss=total_loss):
            #Check the time for each sess run
            start_time = time.time()
            _,total_loss, global_step_count= sess.run([train_op,loss, global_step ])
            time_elapsed = time.time() - start_time
            global_step_count=global_step_count+1
            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss
        #Now finally create all the summaries you need to monitor and group them into one summary op.
        A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        a=tf.placeholder(shape=[],dtype=tf.float32)
        Precision=tf.assign(A, a)
        B = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        b=tf.placeholder(shape=[],dtype=tf.float32)
        Recall=tf.assign(B, b)
        C = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        c=tf.placeholder(shape=[],dtype=tf.float32)
        mIOU=tf.assign(C, c)	
        predictions = tf.argmax(probabilities, -1)
        segmentation_output = tf.cast(tf.reshape((predictions+1)*255/num_classes, shape=[-1, image_height, image_width, 1]),tf.uint8)
        segmentation_ground_truth = tf.cast(tf.reshape(tf.cast(annotations, dtype=tf.float32)*255/num_classes, shape=[-1, image_height, image_width, 1]),tf.uint8)		
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/Precision', Precision)
        tf.summary.scalar('Monitor/Recall_rate', Recall)
        tf.summary.scalar('Monitor/mIoU', mIOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/original_image', realimg, max_outputs=1)
        tf.summary.image('Images/segmentation_output', segmentation_output, max_outputs=1)
        tf.summary.image('Images/segmentation_ground_truth', segmentation_ground_truth, max_outputs=1)
        my_summary_op = tf.summary.merge_all()
        def train_sum(sess, train_op, global_step,sums,loss=total_loss,pre=0,recall=0,iou=0):
            start_time = time.time()
            _,total_loss, global_step_count,ss = sess.run([train_op,loss, global_step,sums ],feed_dict={a:pre,b:recall,c:iou})
            time_elapsed = time.time() - start_time
            global_step_count=global_step_count+1
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss,ss
			
        def eval_step(sess,i ):
            and_eval_batch,T_eval_batch,or_eval_batch,R_eval_batch = sess.run([and_sum,or_sum,apors,R_sum])
            #Log some information
            logging.info('STEP: %d ',i)
            return  and_eval_batch,T_eval_batch,or_eval_batch,R_eval_batch
        def eval(num_class,csvname,session,image_val,eval_batch):
            or_=np.zeros((num_class), dtype=np.float32)
            and_=np.zeros((num_class), dtype=np.float32)			
            T_=np.zeros((num_class), dtype=np.float32)			
            R_=np.zeros((num_class), dtype=np.float32)			
            for i in range(math.ceil(len(image_val) / eval_batch)):
                and_eval_batch,T_eval_batch,or_eval_batch,R_eval_batch = eval_step(session,i+1)
                and_=and_+and_eval_batch
                or_=or_+or_eval_batch
                T_=T_+T_eval_batch
                R_=R_+R_eval_batch				
            Recall_rate=and_/T_
            Precision=and_/R_
            IoU=and_/or_
            mPrecision=np.mean(Precision)
            mRecall_rate=np.mean(Recall_rate)
            mIoU=np.mean(IoU)
            print("Precision:")
            print(Precision)
            print("Recall rate:")
            print(Recall_rate)
            print("IoU:")
            print(IoU)
            print("mPrecision:")
            print(mPrecision)
            print("mRecall_rate:")
            print(mRecall_rate)
            print("mIoU")
            print(mIoU)
            with open(csvname,'a', newline='') as out:
                csv_write = csv.writer(out,dialect='excel')
                csv_write.writerow(Precision)
                csv_write.writerow(Recall_rate)
                csv_write.writerow(IoU)
            return mPrecision,mPrecision,mIoU
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        init = tf.global_variables_initializer()
        saver=tf.train.Saver(max_to_keep=10)
        with tf.Session(config=config) as sess:
            sess.run(init)
            sess.run([titerator.initializer,viterator.initializer])
            step = 0;
            if Start_train is not True:
                #input the checkpoint address,and the step number.
                checkpoint='./log/erfpspial/model.ckpt-37127'
                saver.restore(sess, checkpoint)
                step = 37127
                sess.run(tf.assign(global_step,step))
            summary_writer = tf.summary.FileWriter(logdir, sess.graph)
            final = num_steps_per_epoch * num_epochs
            for i in range(step,final,1):
                if i % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', i/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    if i is not step:
                        saver.save(sess, os.path.join(logdir,log_name),global_step=i)					
                        mPrecision,mRecall_rate,mIoU=eval(num_class=num_classes,csvname=csvname,session=sess,image_val=image_val_files,eval_batch=eval_batch_size)                       				
                if i % min(num_steps_per_epoch, 10) == 0:
                    loss,summaries = train_sum(sess, train_op,global_step,sums=my_summary_op,loss=total_loss,pre=mPrecision,recall=mPrecision,iou=mIoU)
                    summary_writer.add_summary(summaries,global_step=i+1)
                else:
                    loss = train_step(sess, train_op, global_step)
            summary_writer.close()					
            eval(num_class=num_classes,csvname=csvname,session=sess,image_val=image_val_files,eval_batch=eval_batch_size)
            logging.info('Final Loss: %s', loss)
            logging.info('Finished training! Saving model to disk now.')
            saver.save(sess,  os.path.join(logdir,log_name), global_step = final)


if __name__ == '__main__':
    run()
