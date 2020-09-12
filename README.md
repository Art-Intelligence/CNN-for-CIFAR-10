# CNN-for-CIFAR-10

The final version of CNN has 12 convolutional layers, organized into 4 groups of 3 CONV2D layers, each followed by a pooling layer and batch normalization layer. The depth of each group of convolutional layers was increased from 64 to 512. No fully connected hidden layer was implemented for the final version. 

The training loss is diminished by training through epoches but the error rate of classifying the testing data is capped around 15% (by the end of epoch 50) :


epochs, training_loss, testing_error_rate
 
 2, 0.856, 0.29

 4, 0.595, 0.23
 
 6, 0.442, 0.20
 
 8, 0.343, 0.19
 
10, 0.292, 0.19

25, 0.114, 0.16

50, 0.011, 0.15


The CNNs in this repo are run locally by CUDA on NVidia GTX 1070. The average running time training the final version of CNN through each epoch is 92.70 seconds, in contrast with 25.68 seconds for the baseline CNN.

Main body of CNN codes were completed before September 11, 2020. Supplementary jupyter notebook contents were added on September 12, 2020 for visualizations and working towards an assignment paper. An NN-SVG style illustration of the final CNN implemented is attached.
