# Tensorflow Petals Around The Rose
This is a project I created as an introduction to tensorflow.

To get this code to run, you need tensorflow installed. I made this code for python3.

    python3 petals.py
    
or 

    python3 petals_longshot.py

# Petals Around the Rose
Petals Around the Rose is a mathematical logic/puzzle game. To play the game, you roll five dice and ask someone who has already figured out the puzzle what the score is for that collection of dice. You have solved the puzzle when you have figured out the algorithm that converts the 5 dice inputs into the correct score.

When I was first introduced to the puzzle, it took me a couple of days to figure out. After solving it, I have created a program that learns the solution to the puzzle from a dataset of dice rolls and scores.

Check out [this](http://www.borrett.id.au/computing/petals-j.htm) link for an in-depth introduction to the puzzle.

# Results
The programs I created maps the dice rolled to inputs for an artificial sigmoid neuron then uses gradient decent to optimize the weights and bias for the artificial neuron.

 petals.py can learn the relationship between inputs and the score with 100% accuracy.
 
 petals_longshot.py, which does not rearrange the input data at all, can reach an accuracy of 80%.

