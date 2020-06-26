# Flower Image Classifier Project
  (Dataset not available here due to large size)
  
## Argument Parser
### Training
In training one can pass following arguments to train a model differently
 1. No.of Epochs --epochs
 2. Enabling GPU --gpu
 3. Learning Rate --learning_rate
 4. Architecture(like vgg16,alexenet etc...) --arch
 5. Name of checkpoint file --save_dir
 6. Similarly for no. of hidden layers , dropout......
    
### Prediction
During prediction one can choose from following functionalities
 1.For specific image input to model --input_img **(Specify full image path for correct prediction)**
 2.To load a specific checkpoint --checkpoint etc...
