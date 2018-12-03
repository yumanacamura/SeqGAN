import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import pickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 64 # embedding dimension
HIDDEN_DIM = 64 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 2000
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
generated_num = 1024


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file,kigo_list):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def generate_samples_with_pred(sess, generator, discriminator, batch_size, generated_num, output_file,kigo_list):
    # Generate Samples & Predict Samples
    generated_samples = []
    predictions = []
    for _ in range(int(generated_num / batch_size)):
        samples = generator.generate(sess)
        generated_samples.extend(samples)
        feed = {
            discriminator.input_x: samples,
            discriminator.dropout_keep_prob: 1
        }
        new_predictions = sess.run(discriminator.ypred_for_auc, feed_dict = feed)
        predictions.extend(new_predictions)

    data = []
    for h,p in zip(generated_samples,predictions):
        tmp = '{0},{1}'.format(
            ' '.join([str(i) for i in h]),
            p[1]
        )
        data.append(tmp)
    with open(output_file, 'w') as f:
        f.write('\n'.join(data))

def select_haikus(haiku_list, selected_num, output_file):
    # Select Haikus
    idx = np.random.randint(0, haiku_list.shape[0], selected_num)

    haiku = haiku_list[idx]

    with open(output_file, 'w') as fout:
        for poem in haiku:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def pre_train_epoch(sess, trainable_model, data_loader, kigo_list):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    with open('data/ihaiku.pickle','rb') as f:
        haiku_list = pickle.load(f)
    #usew2v---------------------------------------------------------------------------------------------
    with open('data/index.pickle','rb') as f:
        index = pickle.load(f)
    vocab_size = len(index)
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        select_haikus(haiku_list, generated_num, positive_file)
        gen_data_loader.create_batches(positive_file)
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            print('pre-train epoch ', epoch)
            buffer = 'epoch:\t'+ str(epoch) + '\n'
            log.write(buffer)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(50):
        select_haikus(haiku_list, generated_num, positive_file)
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            kigos = select_kigos(kigo_list, BATCH_SIZE)
            samples, rate = generator.generate_with_rate(sess,kigos)
            rewards = rollout.get_reward(sess, samples, 16, discriminator, rate)
            feed = {
                generator.x: samples,
                generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            select_haikus(haiku_list, generated_num, positive_file)
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

        # Test
        print('total_batch:', total_batch,)
        if total_batch-1 % 50 == 0:
            output_file = 'result/result_{0:04d}_epoch.txt'.format(total_batch)
            generate_samples_with_pred(sess, generator, discriminator, BATCH_SIZE, generated_num, output_file)
            buffer = 'epoch:\t' + str(total_batch) + '\n'
            log.write(buffer)

    log.close()


if __name__ == '__main__':
    main()
