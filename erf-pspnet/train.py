import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import model
import random
import csv
import math
slim = tf.contrib.slim

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/camyuan3', 'The log directory to save your checkpoint and event files.')
#Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 8, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 24, 'The batch size used for validation.')
flags.DEFINE_integer('image_height',360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 1e-3, 'The initial learning rate for your training.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
eval_batch_size = FLAGS.eval_batch_size #Can be larger than train_batch as no need to backpropagate gradients.


#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8

#Use median frequency balancing or not

#Visualization and where to save images



#Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

#===============DATASET FOR TRAINING AND EVALATING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, 'train', file) for file in os.listdir(dataset_dir + "/train") if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, "trainannot", file) for file in os.listdir(dataset_dir + "/trainannot") if file.endswith('.png')])

image_val_files = sorted([os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "/val") if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "/valannot") if file.endswith('.png')])



#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = math.ceil(len(image_files) / batch_size)
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#=================CLASS WEIGHTS===============================

class_weights=np.array([ 6.10711717,  4.57716287, 42.28705255,  3.46893819, 16.45916311,  9.60914246, 33.93236668, 33.06296333, 13.5811212 , 40.96211531,44.98280801, 0], dtype=np.float32)

#=================DATA AUGUMENTATION WILL BE UPDATED IN THE FUTURE===============================

def weighted_cross_entropy(onehot_labels, logits, class_weights):
    a=tf.reduce_sum(-tf.log(tf.clip_by_value(logits, 1e-10, 1.0))*onehot_labels*class_weights)
    return a


def decode(a,b):
    a = tf.read_file(a)
    a=tf.image.decode_png(a, channels=3)
    a = tf.image.convert_image_dtype(a, dtype=tf.float32)
    b = tf.read_file(b)
    b = tf.image.decode_png(b,channels=1)
    bb,bs,_=tf.image.sample_distorted_bounding_box(tf.shape(b),bounding_boxes=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4]),min_object_covered=1)
    a=tf.slice(a,bb,bs)
    b=tf.slice(b,bb,bs)
    a=tf.image.resize_images(a, [image_height,image_width],method=0)
    b=tf.image.resize_images(b, [image_height,image_width],method=1)   
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width,1))
    return a,b


def decodev(a,b):
    a = tf.read_file(a)
    a=tf.image.decode_image(a, channels=3)
    a = tf.image.convert_image_dtype(a, dtype=tf.float32)
    b = tf.read_file(b)
    b = tf.image.decode_image(b)
    bb,bs,_=tf.image.sample_distorted_bounding_box(tf.shape(b),bounding_boxes=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4]),min_object_covered=1.0)
    a=tf.slice(a,bb,bs)
    b=tf.slice(b,bb,bs)
    a=tf.image.resize_images(a, [image_height,image_width],method=0)
    b=tf.image.resize_images(b, [image_height,image_width],method=1)
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width,1))
    return a,b
	

with  open('camyuan3.csv','a', newline='') as out:
    csv_write = csv.writer(out,dialect='excel')
    a=[str(i) for i in range(num_classes)]
    csv_write.writerow(a)
    



	
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


        logits, probabilities= model.train(images,numclasses=num_classes, shape=[image_height,image_width], l2=weight_decay,reuse=None,is_training=True)
        annotations = tf.reshape(annotations, shape=[-1, image_height, image_width])
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)
        predictions = tf.argmax(probabilities, -1)
        segmentation_output = tf.cast(predictions, dtype=tf.float32)
        segmentation_output = tf.reshape(segmentation_output, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth = tf.cast(annotations, dtype=tf.float32)
        segmentation_ground_truth = tf.reshape(segmentation_ground_truth, shape=[-1, image_height, image_width, 1])
        YANMO = tf.reduce_sum(1-annotations_ohe[:,:,:,-1])
		
        #Actually compute the loss

        los = weighted_cross_entropy(logits=probabilities, onehot_labels=annotations_ohe, class_weights=class_weights)/YANMO
        loss=tf.losses.add_loss(los)
        total_loss = tf.losses.get_total_loss()

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()








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
            train_op = slim.learning.create_train_op(total_loss, optimizer)

        predictions = tf.argmax(probabilities, -1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)
        metrics_op = tf.group(accuracy_update, mean_IOU_update)

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_val, mean_IOU_val, _ = sess.run([train_op, global_step, accuracy, mean_IOU, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f', global_step_count, total_loss, time_elapsed, accuracy_val, mean_IOU_val)

            return total_loss, accuracy_val, mean_IOU_val

        #================VALIDATION BRANCH========================
        #Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        vdataset = tf.data.Dataset.from_tensor_slices((images_val,annotations_val))
        vdataset = vdataset.map(decodev)
        vdataset = vdataset.batch(eval_batch_size).repeat(num_epochs+5)
        viterator = vdataset.make_initializable_iterator()
        images_val,annotations_val = viterator.get_next()		
        logits_val, probabilities_val= model.train(images_val,numclasses=num_classes, shape=[image_height,image_width], l2=weight_decay,reuse=True,is_training=False)
        annotations_val = tf.reshape(annotations_val, shape=[-1, image_height, image_width])
        annotations_ohe_val = tf.one_hot(annotations_val, num_classes, axis=-1)

		
        predictions_val = tf.argmax(probabilities_val, -1)
        predictions_vals = tf.one_hot(predictions_val, num_classes, axis=-1)

		
		
		
        apand=annotations_ohe_val*predictions_vals
        apands=tf.reduce_sum(apand,[0,1,2])
        nor = tf.reshape((1-annotations_ohe_val[:,:,:,-1]),shape=[-1,image_height,image_width,1])
        apor=tf.to_int32((annotations_ohe_val+predictions_vals)*nor>0.5)
        apors=tf.reduce_sum(apor,axis=[0,1,2])
        aptrue=tf.reduce_sum(annotations_ohe_val,axis=[0,1,2])
        A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        a=tf.placeholder(shape=[],dtype=tf.float32)
        mean_IOU_val=tf.assign(A, a)
        vali_classiou=0.0;


        def eval_step(sess,i ):

            ands,trues,ors = sess.run([apands,aptrue,apors])

            #Log some information
            logging.info('STEP: %d ',i)

            return  ands,trues,ors

        #=====================================================

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/original_image', images, max_outputs=1)
        tf.summary.image('Images/segmentation_output', segmentation_output, max_outputs=1)
        tf.summary.image('Images/segmentation_ground_truth', segmentation_ground_truth, max_outputs=1)
        my_summary_op = tf.summary.merge_all()

        def train_sum(sess, train_op, global_step, metrics_op,sums,shu):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_val, mean_IOU_val,ss, _ = sess.run([train_op, global_step, accuracy, mean_IOU,sums ,metrics_op],feed_dict={a:shu})
            time_elapsed = time.time() - start_time

            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f', global_step_count, total_loss, time_elapsed, accuracy_val, mean_IOU_val)

            return total_loss, accuracy_val, mean_IOU_val,ss
        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=None)
        # Run the managed session
        with sv.managed_session() as sess:
            sess.run([viterator.initializer,titerator.initializer])
            ors=np.zeros((num_classes), dtype=np.float32)
            ans=np.zeros((num_classes), dtype=np.float32)
            trues=np.zeros((num_classes), dtype=np.float32)
            for i in range(math.ceil(len(image_val_files) / eval_batch_size)):
                andss,truess,orss = eval_step(sess,i+1)
                ans=ans+andss
                ors=ors+orss
                trues=trues+truess
            vali_iou=ans/ors
            vali_class=ans/trues
            vali_classiou=np.mean(vali_iou[0:-1])
            vali_classavg=np.mean(vali_class[0:-1])
            print(vali_iou)
            print(vali_classiou)
            with open('camyuan3.csv','a', newline='') as out:
                csv_write = csv.writer(out,dialect='excel')
                csv_write.writerow(vali_class[0:-1])
                csv_write.writerow(vali_iou[0:-1])
				
            for step in range(int(num_steps_per_epoch * num_epochs)):
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)

                
                if step % min(num_steps_per_epoch, 10) == 0:
                    loss, training_accuracy, training_mean_IOU,summaries = train_sum(sess, train_op, sv.global_step, metrics_op=metrics_op,sums=my_summary_op,shu=vali_classiou)
                    sv.summary_computed(sess, summaries)
 
                else:
                    loss, training_accuracy,training_mean_IOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

					
					
                if (step+1) % (num_steps_per_epoch ) == 0:
                    ors=np.zeros((num_classes), dtype=np.float32)
                    ans=np.zeros((num_classes), dtype=np.float32)
                    trues=np.zeros((num_classes), dtype=np.float32)
                    for i in range(math.ceil(len(image_val_files) / eval_batch_size)):
                        andss,truess,orss = eval_step(sess,i+1)
                        ans=ans+andss
                        ors=ors+orss
                        trues=trues+truess
                    vali_iou=ans/ors
                    vali_class=ans/trues
                    vali_classiou=np.mean(vali_iou[0:-1])
                    vali_classavg=np.mean(vali_class[0:-1])
                    print(vali_iou)
                    print(vali_classiou)
                    with open('camyuan3.csv','a', newline='') as out:
                        csv_write = csv.writer(out,dialect='excel')
                        csv_write.writerow(vali_class[0:-1])
                        csv_write.writerow(vali_iou[0:-1])
            logging.info('Final Loss: %s', loss)
            logging.info('Final Training Accuracy: %s', training_accuracy)
            logging.info('Final Training Mean IOU: %s', training_mean_IOU)
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    run()
