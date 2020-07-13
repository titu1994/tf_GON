# Ported from https://github.com/cwkx/GON/blob/master/GON.py
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tf_siren
import utils

LR = 1e-4
BATCH_SIZE = 144  # needs to be a square value - 64, 144 etc
HIDDEN_DIM = 32
LATENT_DIM = 32
NUM_LAYERS = 4
STEPS = 5000
SAVE_IMAGE_EVERY_N_STEPS = 500

basedir = 'mnist'
image_dir = os.path.join(basedir, 'images')
utils.create_dirs(image_dir)

ds, ds_info = tfds.load('mnist', split='train', with_info=True)  # type: tf.data.Dataset

input_shape = ds_info.features['image'].shape
dataset_len = ds_info.splits['train'].num_examples

rows, cols, channels = input_shape
pixel_count = rows * cols
img_coords = 2


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
model = tf_siren.SIRENModel(units=HIDDEN_DIM, final_units=channels, num_layers=NUM_LAYERS, final_activation='sigmoid')

# Instantiate the model
dummy_input = tf.zeros([BATCH_SIZE, pixel_count, img_coords + LATENT_DIM])
_ = model(dummy_input)

model.summary()


@tf.function
def train_step(train_iterator, model: tf.keras.Model, optimizer):
    # sample a batch of data
    x = next(train_iterator)
    x = tf.reshape(x, [x.shape[0], -1, channels])

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            # compute the gradients of the inner loss with respect to zeros (gradient origin)
            z = tf.zeros([BATCH_SIZE, 1, LATENT_DIM])
            inner_tape.watch(z)

            z_rep = tf.tile(z, (1, c.shape[1], 1))

            g = model(tf.concat((c, z_rep), axis=-1))
            L_inner = tf.reduce_mean(tf.reduce_sum((g - x) ** 2, axis=1))

        z = -inner_tape.gradient(L_inner, z)

        # now with z as our new latent points, optimise the data fitting loss
        z_rep = tf.tile(z, (1, c.shape[1], 1))
        g = model(tf.concat((c, z_rep), axis=-1))
        L_outer = tf.reduce_mean(tf.reduce_sum((g - x) ** 2, axis=1))

    grads = outer_tape.gradient(L_outer, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return g, z, L_outer,


# Train Gradient Origin Network
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
c = tf.stack([utils.get_mgrid(rows, 2) for _ in range(BATCH_SIZE)])  # coordinates
train_iterator = iter(ds)

recent_zs = []
average_loss = tf.keras.metrics.Mean()
for step in range(STEPS + 1):
    g, z, L_outer = train_step(train_iterator, model, optimizer)

    # update matric
    average_loss.update_state(L_outer)

    # compute sampling statistics
    recent_zs.append(z)
    recent_zs = recent_zs[-100:]

    if step % SAVE_IMAGE_EVERY_N_STEPS == 0 and step > 0:
        print(f"Step: {step}   Loss: {average_loss.result().numpy():.3f}")

        # reset metric
        average_loss.reset_states()

        g = tf.clip_by_value(g, 0.0, 1.0)
        g = tf.reshape(g, [-1, rows, cols, channels])

        # plot reconstructions
        utils.save_image(g, f'{image_dir}/recon_{step}.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)

        # plot interpolations
        f = utils.slerp_batch(model, z, c, BATCH_SIZE)
        f = tf.clip_by_value(f, 0.0, 1.0)
        f = tf.reshape(f, [-1, rows, cols, channels])

        utils.save_image(f, f'{image_dir}/slerp_{step}.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)

        # plot samples
        s = utils.gon_sample(model, recent_zs, c, BATCH_SIZE)
        s = tf.clip_by_value(s, 0.0, 1.0)
        s = tf.reshape(s, [-1, rows, cols, channels])
        utils.save_image(s, f'{image_dir}/sample_{step}.png', nrow=int(np.sqrt(BATCH_SIZE)), padding=0)
