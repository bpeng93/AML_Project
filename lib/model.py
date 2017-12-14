
def SVHN(x, n_classes):
    import tools
    '''
    Args:
        images: 4D tensor [batch_size, img_width, img_height, img_channel]
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.


    '''
    x = tools.conv('conv1', x, 64)
    x = tools.pool('pool1', x)

    x = tools.conv('conv2', x, 64)
    x = tools.pool('pool2', x)

    x = tools.conv('conv3', x, 128)
    x = tools.pool('pool3', x)

    x = tools.FC_layer('fc4', x, out_nodes = 64)
    x = tools.drop_out('drop_out', x, keep_prob = 0.5)
    x = tools.final_layer('softmax', x, out_nodes=n_classes)
    return x
