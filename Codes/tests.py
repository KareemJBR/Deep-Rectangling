import tensorflow as tf
from utils import DataLoader
from model import rectangling_network

if __name__ == "__main__":
    train_data_loader = DataLoader('./DIR-D/training/')
    train_data_dataset = train_data_loader(batch_size=1)
    train_data_it = train_data_dataset.make_one_shot_iterator()
    train_input_tensor = train_data_it.get_next()
    train_input_tensor.set_shape([1, 384, 512, 9])

    train_input = train_input_tensor[:, :, :, 0:3]
    train_mask = train_input_tensor[:, :, :, 3:6]
    train_gt = train_input_tensor[:, :, :, 6:9]

    train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final, train_warp_image_final, \
        train_warp_mask_final = rectangling_network(train_input, train_mask)

    tf.summary.image(tensor=train_input, name='train_input')
    tf.summary.image(tensor=train_mask, name='train_mask')
    tf.summary.image(tensor=train_gt, name='train_gt')
    tf.summary.image(tensor=train_warp_image_primary, name='train_warp_image_primary')
    tf.summary.image(tensor=train_warp_image_final, name='train_warp_image_final')
