# Ported from https://github.com/cwkx/GON/blob/master/GON.py
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tf_siren
import utils

BATCH_SIZE = 144  # needs to be a square value - 64, 144 etc
HIDDEN_DIM = 256
LATENT_DIM = 256
NUM_LAYERS = 4

basedir = 'cifar10'
image_dir = os.path.join(basedir, 'images')
weights_dir = os.path.join(basedir, 'weights')

utils.create_dirs(image_dir)
utils.create_dirs(weights_dir)

ds, ds_info = tfds.load('cifar10', split='train', with_info=True)  # type: tf.data.Dataset

input_shape = ds_info.features['image'].shape
dataset_len = ds_info.splits['train'].num_examples

rows, cols, channels = input_shape
pixel_count = rows * cols
img_coords = 2

print("Pixel count :", pixel_count)


def prepare_dataset(ds):
    image = ds['image']
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image


# Dataset preparation

ds = ds.map(prepare_dataset, num_parallel_calls=2 * os.cpu_count())
ds = ds.shuffle(dataset_len)
ds = ds.batch(BATCH_SIZE, drop_remainder=True)
ds = ds.repeat()
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

# Build GON Model (SIREN)

# Train Gradient Origin Network
model = tf_siren.SIRENModel(units=HIDDEN_DIM, final_units=channels, num_layers=NUM_LAYERS,
                            final_activation='sigmoid')

# Instantiate the model
dummy_input = tf.zeros([BATCH_SIZE, pixel_count, img_coords + LATENT_DIM])
_ = model(dummy_input)

# Restore weights
model.load_weights(weights_dir + '/model').assert_consumed()

# Summary of model
model.summary()


@tf.function
def eval_step(train_iterator, model: tf.keras.Model):
    # sample a batch of data
    x = next(train_iterator)
    x = tf.reshape(x, [x.shape[0], -1, channels])

    with tf.GradientTape() as inner_tape:
        # compute the gradients of the inner loss with respect to zeros (gradient origin)
        z = tf.zeros([BATCH_SIZE, 1, LATENT_DIM])
        inner_tape.watch(z)

        z_rep = tf.tile(z, (1, coords.shape[1], 1))

        g = model(tf.concat((coords, z_rep), axis=-1))
        L_inner = tf.reduce_mean(tf.reduce_sum((g - x) ** 2, axis=[1, 2]))

    z = -inner_tape.gradient(L_inner, z)

    # now with z as our new latent points, optimise the data fitting loss
    z_rep = tf.tile(z, (1, coords.shape[1], 1))
    g = model(tf.concat((coords, z_rep), axis=-1))
    L_outer = tf.reduce_mean(tf.reduce_sum((g - x) ** 2, axis=[1, 2]))

    return x, g, z, L_outer,


coords = tf.stack([utils.get_mgrid(rows, 2) for _ in range(BATCH_SIZE)])  # coordinates  [-1, 1]
train_iterator = iter(ds)

recent_zs = []
gt, g, z, L_outer = eval_step(train_iterator, model)

# compute sampling statistics
recent_zs.append(z)
recent_zs = recent_zs[-100:]

print(f"Loss: {L_outer.numpy():.3f}  Per-Channel Loss: {(L_outer.numpy() / channels):.3f}")

gt = tf.reshape(gt, [-1, rows, cols, channels])

g = tf.clip_by_value(g, 0.0, 1.0)
g = tf.reshape(g, [-1, rows, cols, channels])

# plot ground truth
utils.save_image(gt, f'{image_dir}/ground_truth_eval.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)

# plot reconstructions
utils.save_image(g, f'{image_dir}/recon_eval.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)

# plot interpolations
f = utils.slerp_batch(model, z, coords, BATCH_SIZE)
f = tf.clip_by_value(f, 0.0, 1.0)
f = tf.reshape(f, [-1, rows, cols, channels])

utils.save_image(f, f'{image_dir}/slerp_eval.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)

# plot samples
s = utils.gon_sample(model, recent_zs, coords, BATCH_SIZE)
s = tf.clip_by_value(s, 0.0, 1.0)
s = tf.reshape(s, [-1, rows, cols, channels])
utils.save_image(s, f'{image_dir}/sample_eval.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)
