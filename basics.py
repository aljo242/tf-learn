import tensorflow as tf

def printSpacer():
    print(" ")
    print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(" ")

def init():
    printSpacer
    print(tf.version)
    string = tf.Variable("this is a string", tf.string)


def tensors():
    printSpacer()

    # basic data types [string, int*, float*]
    string = tf.Variable("this is a string", tf.string)
    number = tf.Variable(324, tf.int16)
    floating = tf.Variable(3.567, tf.float64)

    # tensor is just some N-dim array 
    rank1Tensor = tf.Variable(["Test"], tf.string)
    rank2Tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

    tf.print(tf.rank(rank1Tensor))
    tf.print(rank1Tensor.shape)

    print(tf.rank(rank2Tensor))
    tf.print(rank2Tensor.shape)

    # modify shapes
    tensor1 = tf.ones([1, 2, 3]) # creates a shape [1, 2, 3] tensor with ones
    tf.print(tensor1)

    tensor2 = tf.reshape(tensor1, [2, 3, 1])
    tf.print(tensor2)

    tensor3 = tf.reshape(tensor2, [3, -1]) # providing -1 arguments means "infer this dimension given the other ones"
    tf.print(tensor3)

    # all main types of tensors
    #   Variable
    #   Constant
    #   PlaceHolder
    #   SparseTensor
    # 
    # NOTE: all are immutable besides Variable

    # evaluating tensors
    #       tensors store graphs of operations, so they are "lazily evaluated"
    #       only once we invoke a session, will we actaully call the computation to be performed
    #       everything before that point is just building up a compute graph to later execute on
    #       If we think about this a bit, this is kind of slick because perhaps some optimizations 
    #       can be made before we "eagerly" execute everything
  


def main():
    init()
    tensors()


if __name__ == "__main__":
    main()