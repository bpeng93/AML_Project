# SVHN Final Model

## Model Design

In the first part, we find the best model by achiving higher accuracy rate on **3** classes.

original images are single digit 3 channels 32 x 32 images. 

finding the best images we only used training data set from the 
[SVHN webset](http://ufldl.stanford.edu/housenumbers/), because it is enought data to train out network. The training data has 73257 record. 



**4** python files in choosing the best model. 

`input_data.py` contains 2 functions `GCN` and `reas_SVHN`.
`GCN` first transpose the original color images to gray scale images then perform `global_contrast_normalize` which subtracts the mean across features and then normalizes standard deviation (across features, for each example). `GCN` also randamly divides the given data into train and validation part. The returns of `GCN` are training and validation set with the size of number of records x 1025. Each record has 1 label bytes plus 32*32 image bytes. 

`read_SVHN` reads numpy array data and generates batches for tensorflow. Tensorflow takes the output of `read_SVHN`. 


`tools.py` defines building block for the structure of the network. `conv` is the convolution layer, `pool`  is the pooling layer, FC_layer and final_layer are the full conection layers, and drop_out is the drop out layer. when re-design the network only need to change the    kernel size or  stride the default values are given but able to change them in the `model.py`. `loss`, `accuracy` and `optimize` metric or method to get the best model. 

`model.py` has the function `SVHN` function which defines the struture of the model **3** convolution layers and **2** full connection layers. Each convolution layers is followed by a pooling layer and the drop out layer is in the middle of **2** convolution layers. 

## Model Performance






 
