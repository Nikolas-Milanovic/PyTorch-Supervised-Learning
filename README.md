# PyTorch-Supervised-Learning
Neural Network - comparing Architecture, Activation Functions, Dropout comparison

# Results

## Architecture comparison:

<img width="228" alt="image" src="https://github.com/Nikolas-Milanovic/PyTorch-Supervised-Learning/assets/59632554/b4eb7e6f-8e41-494a-9184-ad52782fac60">
<img width="229" alt="image" src="https://github.com/Nikolas-Milanovic/PyTorch-Supervised-Learning/assets/59632554/c3d2e397-16d5-48c2-ab33-88442b3dc9f5">


From the graphs above we can see the as we add additional layers to the neural network, the training accuracy improves with the number of epoch. This makes sense as when we add more layers the network we have the ability to learn more. However, we learn too much! The test accuracy shows that the networks with more layers learnt too much, as a result overfitted. This overfitting negatively affected the testing accuracy as we learnt more (increased epoch).


## Activation function comparison:

<img width="236" alt="image" src="https://github.com/Nikolas-Milanovic/PyTorch-Supervised-Learning/assets/59632554/211ac3e1-3bd8-4d23-83d5-d2be73785c50">
<img width="226" alt="image" src="https://github.com/Nikolas-Milanovic/PyTorch-Supervised-Learning/assets/59632554/b8875aa2-c7f6-4f98-8958-4f75b739919a">

ReLU here is better than sigmoid for its ability to mitigate the vanishing gradient problem. Sigmoid converts any value of range of [0,1] and normalizes to zero or one. Therefore sigmoid losses more information and hence learns slower, as seen in the training accuracy. Moreover, in this network, the sigmoid will shrink the values and when this is multiplied the inputs, it’s not going to have as large of an effect as a ReLU where large/positive values stay large. So that’s why ReLU gets trained faster because the values are not restricted. Overall, the ReLU learns faster but this excess learning results in overfitting as the epoch increases. The same overfitting eventually happens to Sigmoid, but with a greater number of epochs. 

 
## Dropout comparison:

<img width="233" alt="image" src="https://github.com/Nikolas-Milanovic/PyTorch-Supervised-Learning/assets/59632554/bf2a05a0-8b48-4624-9256-bdc1534a2d04">
<img width="233" alt="image" src="https://github.com/Nikolas-Milanovic/PyTorch-Supervised-Learning/assets/59632554/8a9ab373-f536-413c-a70f-673316964bba">

Here we see that adding a dropout improves the testing accuracy. This is because dropout works by randomly disabling a fraction of neurons during training. This helps prevent overfitting by promoting redundancy and reducing the network's reliance on specific neurons, leading to improved generalization and robustness.



