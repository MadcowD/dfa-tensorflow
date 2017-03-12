"""
Tensorflow implementation of direct feedback alignement.
Author: William H. Guss.
"""
import operator
import tensorflow as tf
from functools import reduce

def get_num_nodes_respect_batch(tensor_shape):
    shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0
    return reduce(operator.mul, tensor_shape.as_list()[shape_start_index:], 1), shape_start_index
    
def random_matrix(shape):
    with tf.variable_scope("radom_matrix"):
        rand_t = tf.random_uniform(shape, -1, 1)
        return tf.Variable(rand_t, name="weights")

def flatten_respect_batch(tensor):
    """
    Flattens a tensor respecting the batch dimension.
    Returns the flattened tensor and its shape as a list. Eg (tensor, shape_list).
    """
    with tf.variable_scope("flatten_respect_batch"):
        shape = tensor.get_shape()
        num_nodes, shape_start_index = get_num_nodes_respect_batch(shape)

        # Check if the tensor is already flat!
        if len(shape) - shape_start_index == 1:
            return tensor, shape.as_list()

        # Flatten the tensor respecting the batch.
        if shape_start_index > 0:
            flat = tf.reshape(tensor, [-1, num_nodes])
        else:
            flat = tf.reshape(tensor, [num_nodes])

        return flat

def reshape_respect_batch(tensor, out_shape_no_batch_list):
    """
    Reshapes a tensor respecting the batch dimension.
    Returns the reshaped tensor
    """
    with tf.variable_scope("reshape_respect_batch"):
        tensor_shape = tensor.get_shape()
        shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0

        # Flatten the tensor respecting the shape.
        if shape_start_index > 0:
            shaped = tf.reshape(tensor, [-1] + out_shape_no_batch_list)
        else:
            shaped = tf.reshape(tensor, out_shape_no_batch_list)

        return shaped

def direct_feedback_alignement(optimizer, loss, output, activation_param_pairs):
    """
    Builds a series of gradient ops which constitute direct_feedback_alignment.
    Params:
        - OPTIMIZER: A tf.train.Optimizer to apply to the direct feedback. Eg. tf.train.AdamOptimizer(1e-4)
        - LOSS: A loss function of the OUTPUTs to optimize.
        - OUTPUT: An output tensor for wahtever tensorflow graph we would like to optimize.
        - ACTIVATION_PARAM_PAIRS: A list of pairs of output activations for every "layer" and the associated weight variables.

    Returns: a training operation similar to OPTIMIZER.minimize(LOSS).
    """
    with tf.variable_scope("direct_feedback_alignment"):
        # Get flatten size of outputs
        out_shape = output.get_shape()
        out_num_nodes, shape_start_index = get_num_nodes_respect_batch(out_shape)
        out_non_batch_shape = out_shape.as_list()[shape_start_index:]

        # Get the loss gradients with respect to the outputs.
        loss_grad = tf.gradients(loss, output)
    
        virtual_gradient_param_pairs = []
        # Construct direct feedback for each layer
        for i, (layer_out, layer_weights) in enumerate(activation_param_pairs):
            with tf.variable_scope("virtual_feedback_{}".format(i)):
                if layer_out is output:
                    proj_out = output
                else:
                    # Flatten the layer (this is naiive with respect to convolutions.)
                    flat_layer, layer_shape = flatten_respect_batch(layer_out)
                    layer_num_nodes = layer_shape[-1]

                    # First make random matrices to virutally connect each layer with the output.
                    rand_projection = random_matrix([layer_num_nodes, out_num_nodes])
                    flat_proj_out = tf.matmul(flat_layer, rand_projection)

                    # Reshape back to output dimensions and then get the gradients.
                    proj_out  = reshape_respect_batch(flat_proj_out, out_non_batch_shape)
                for weight in layer_weights:
                    print(loss_grad, proj_out)
                    virtual_gradient_param_pairs +=  [
                        (tf.gradients(proj_out, weight, grad_ys=loss_grad)[0], weight)]

        train_op = optimizer.apply_gradients(virtual_gradient_param_pairs)
        return train_op 
