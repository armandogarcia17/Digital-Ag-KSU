function net = cscnnBuild(inputSize,numClasses)
%Generates 
layers = [
    image3dInputLayer(inputSize,Name="Input",Normalization="None")
    convolution3dLayer([3 3 7],8,Name="conv3d_1")
    reluLayer(Name="Relu_1")
    convolution3dLayer([3 3 5],16,Name="conv3d_2")
    reluLayer(Name="Relu_2")
    convolution3dLayer([3 3 3],32,Name="conv3d_3")
    reluLayer(Name="Relu_3")
    convolution3dLayer([3 3 1],8,Name="conv3d_4")
    reluLayer(Name="Relu_4")
    fullyConnectedLayer(256,Name="fc1")
    reluLayer(Name="Relu_5")
    dropoutLayer(0.4,Name="drop_1")
    fullyConnectedLayer(128,Name="fc2")
    dropoutLayer(0.4,Name="drop_2")
    fullyConnectedLayer(numClasses,Name="fc3")
    softmaxLayer(Name="softmax")];
net = dlnetwork(layers);
end