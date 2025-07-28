# Learning-Not-to-Learn
An implementation of the paper "Learning Not to Learn" on colored MNIST dataset which tries to make the model unlearn the bias in the training data to increase its accuracy on testing data which has no bias
> Sample of Input
<img width="368" height="372" alt="image" src="https://github.com/user-attachments/assets/3f952fa1-46eb-44f8-8279-05e0097b4767" />

I was able to increase the accuracy of the model from 68% to 90%

## Procedure
1. I began first by creating the datasets
    - Train dataset: every digit has mean color.
    - Test dataset: each sample color is chosen randomly.
2. I created the model as specified in paper
3. Finally I created the loss functions and started training, in the process I was tuning the hyperparameters.

### Confusion Matrix of Base Model
<img width="800" height="717" alt="image" src="https://github.com/user-attachments/assets/031cd9f0-d75d-441f-a2c6-ad62f833427f" />

### Confustion Matrix of Paper Model
<img width="813" height="696" alt="image" src="https://github.com/user-attachments/assets/ad6f5988-fc08-4d7b-84fb-a713be52d852" />


