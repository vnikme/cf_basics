#!/usr/bin/python
# coding: utf-8

from matplotlib import pyplot
import numpy as np
import time, random
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

def PrintBlobs():
    for blob in workspace.Blobs():
        print("{}:\n{}".format(blob, workspace.FetchBlob(blob)))
    print("")

def InitAndCreateNets(batch_size, input_size, layer_sizes, output_size):
    init_net = core.Net("init")
    w, w_m, b, b_m = [], [], [], []
    n = len(layer_sizes)
    for i in xrange(n + 1):
        w_shape = [layer_sizes[i] if i < n else output_size, layer_sizes[i - 1] if i > 0 else input_size]
        w.append(init_net.UniformFill([], ["w%d" % i], min = -0.1, max = 0.1, shape = w_shape))
        w_m.append(init_net.ConstantFill([], ["w%d_m" % i], shape = w_shape, value = 0.0))
        b.append(init_net.UniformFill([], ["b%d" % i], min = -0.1, max = 0.1, shape = [w_shape[0]]))
        b_m.append(init_net.ConstantFill([], ["b%d_m" % i], shape = [w_shape[0]], value = 0.0))
    x = init_net.ConstantFill([], ["x"], shape = [batch_size, input_size], value = 0.0)
    y = init_net.ConstantFill([], ["y"], shape = [batch_size, output_size], value = 0.0)
    it = init_net.ConstantFill([], "it", shape=[1], value=0, dtype=core.DataType.INT32)
    one = init_net.ConstantFill([], "one", shape=[1], value=1.0)
    negone = init_net.ConstantFill([], "negone", shape=[1], value=-1.0)
    workspace.RunNetOnce(init_net) 

    train_net = core.Net("train")
    s = train_net.StopGradient([x], [x])
    y = train_net.StopGradient([y], [y])
    for i in xrange(n + 1):
        l = train_net.FC([s, w[i], b[i]], ["l%d" % i])
        if i < n:
            # Play with nonlinearity used: try Relu and Sigmoid
            s = train_net.Softplus([l], ["s%d" % i])
        else:
            s = train_net.Sigmoid([l], ["s%d" % i])

    dist = train_net.SquaredL2Distance([s, y], "dist")
    loss = train_net.AveragedLoss([dist], ["loss"])
    gradient_map = train_net.AddGradientOperators([loss])
    train_net.Iter(it, it)
    lr = train_net.LearningRate(it, "lr", base_lr = 0.1, policy = "step", stepsize = 10000, gamma = 0.95)

    for i in xrange(n + 1):
        train_net.MomentumSGD([gradient_map[w[i]], w_m[i], lr], [gradient_map[w[i]], w_m[i]], momentum = 0.9, nesterov = 1)
        train_net.WeightedSum([w[i], one, w_m[i], negone], w[i])
        train_net.MomentumSGD([gradient_map[b[i]], b_m[i], lr], [gradient_map[b[i]], b_m[i]], momentum = 0.9, nesterov = 1)
        train_net.WeightedSum([b[i], one, b_m[i], negone], b[i])

    workspace.CreateNet(train_net)
    return train_net


def main():
    # Play with layers sizes and number of layers!
    # For example, 2 layers of 15 neurons converge by 1.5K iterations,
    # while 5 layers of 15 neurons need 200K iterations
    layer_sizes = [15 for i in xrange(5)]
    train_net = InitAndCreateNets(4, 2, layer_sizes, 3)
    data_x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype = np.float32)
    data_y = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype = np.float32) 
    workspace.FeedBlob("x", data_x)
    workspace.FeedBlob("y", data_y)
    i = 0
    while True:
        workspace.RunNet(train_net.Proto().name)
        l = workspace.FetchBlob("loss")
        print("Iteration {}, loss={}\noutput:\n{}\n".format(i, l, "\n".join(map(lambda x: "\t".join(map(lambda y: "%.3f" % y, x)), workspace.FetchBlob("s%d" % len(layer_sizes))))))
        if l < 0.0001:
            break
        i = i + 1
        #if i == 10:
        #    break
        #PrintBlobs()

if __name__ == "__main__":
    main()

