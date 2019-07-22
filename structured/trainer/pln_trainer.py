from base.base_train import BaseTrain
import os
from tqdm import tqdm


class ProtLigNet(BaseTrain):

    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)
    

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))

    
    def train_step(self):


def train():
    my_path = "chimera/preprocessed_data/"
    dataset_path = os.path.join(os.pardir, my_path)
    fill_data(dataset_path)
    graph = make_network(batch_size = 10)
    global_step = graph.get_tensor_by_name("training/global_step:0")
    x = graph.get_tensor_by_name('input/data_x:0')
    t = graph.get_tensor_by_name('input/data_affinity:0')
    y = graph.get_tensor_by_name('output/prediction:0')
    prediction = graph.get_tensor_by_name('output/prediction:0')
    train = graph.get_tensor_by_name('training/train:0')
    mse = graph.get_tensor_by_name('training/mse:0')
    cost = graph.get_tensor_by_name('training/loss:0')

    w_conv0 = graph.get_tensor_by_name('convolution/conv_layer_0/w:0')

    '''
        common summary of the training process


    '''

    common_summary = tf.summary.merge((
        tf.summary.histogram("weights", w_conv0),
        tf.summary.scalar("mse", mse),
        tf.summary.scalar("loss", cost)
    ))
    
    num_epochs = 20
    compare_error = float('inf')

    """
        Some setup for the gpu usage
    """

    config = tf.ConfigProto()
#    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    with graph.as_default():
        saver = tf.train.Saver(max_to_keep = 10)
    

    with tf.Session(graph = graph, config = config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            x_t, y_t = shuffle(range(dset_sizes['training']), affinity['training'])
            
            training_mse_err = 0
            validation_mse_err = 0

            for bi, bj in batches('training'):
                weight = (bj - bi)/dset_sizes["training"]
                session.run(train, feed_dict = {x: g_batch('training', x_t[bi:bj]), t: y_t[bi:bj]})
                train_err = session.run(mse, feed_dict = {x: g_batch('training', x_t[bi:bj]), t: y_t[bi:bj]})
                training_mse_err += weight * train_err

            # summary per each epoch
            training_summary = session.run(common_summary, feed_dict = {x: g_batch('training'), t: y_t[:10]} )
            train_writer.add_summary(training_summary, global_step.eval())

            x_v, y_v = shuffle(range(dset_sizes['validation']), affinity['validation'])

            for bi, bj in batches('validation'):
                weight = (bj - bi)/dset_sizes["validation"]
                val_err = session.run(mse, feed_dict = {x: g_batch('validation', x_v[bi:bj]), t: y_v[bi:bj]})
                validation_mse_err += weight * val_err 

            validation_summary = session.run(common_summary, feed_dict = {x: g_batch('validation'), t: affinity['validation'][:10]} )
            val_writer.add_summary(training_summary, global_step.eval())

            if validation_mse_err < compare_error:
                compare_error = validation_mse_err
                saver.save(session, 'saved_models/model')

            logger.info("AFTER THE EPOCH {}, the mse_training is {} and mse_validation is {}".format(epoch, training_mse_err, validation_mse_err))
