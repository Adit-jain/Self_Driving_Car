# Self_Driving_Car
A virtual car which uses neural networks to determine the path from the top left corner of the screen to the bottom right corner of the screen, to and fro, avoiding any obstacle made by the operator.

The black screen is the area available to explore. Sand is activated by left mouse click and is of yellow colour. The aim of the car is to avoid sand and reach its endpoints as quickly as possible. The car achieves this with the help of neural networks and Deep Q-learning. Q values are calculated for each action via the neural networks and the best action is provided for the car in order to achieve the goal.

The algo uses rectifier activation function and Adam optimizer. The GUI is implemented using Kivy. AN option to save the neural network state i.e brain is available, similarly, one can load the brain to continue from last checkpoint.
