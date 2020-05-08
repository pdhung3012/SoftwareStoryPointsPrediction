# Software Effort Prediction in Agile Development withMachine Learning & Natural Language Processing
The success of software development projects hinges upon, amongst other factors, on project and time management. One popular method used to aid time management and estimating project timelines for agile software development is the estimation number of story points, which represent the amount of development effort per individual software issues or requests in number of man hours. In this paper, we explore various text vectorization machine learning techniques to predict software development effort measured in number of story points. Our results show that the problem can be formulated as both a classification or a regression problem, and successfully solved using supervised learning. Moreover, several of our regression models achieve better accuracy than prior literature. We also demonstrate that Generative Adversarial Network (GAN) semi-supervised yields significantly better results than normal semi-supervised learning. Finally, we show that deep learning architectures such as convolutional attention recurrent neural networks yield very promising results, but require further hyperparameter tuning.

**Authors: Hung Phan, Eliska Koberdanz, Jeremiah Roghair** from Department of Computer Science, Iowa State University.

# Requirements:
- Python 3.6.
- Scikit-learn 0.22.2.
- Keras 2.2.5.
- PyTorch 1.5

# Instruction for Replication:

**RQ1. Classification: What is the accuracy of story points categories classification?**

**RQ2. Regression: What is the accuracy of story points prediction?**

**RQ3. Semi-supervised training: How does the semi-supervised training affect accuracy?**

**RQ4. Hyperparameter Tuning: Does hyperparameter tuning improve the accuracy?**

**RQ5. Vectorization: What text vectorization techniques are most suitable?**

**RQ6. Are deep learning models suitable for software effort estimation?**

1.  Go to  the repo here:  https://github.com/jroghair/Software-Point-Estimation/tree/master/Deep%20Learning%20Models
2.  Open the terminal on windows (this code assumes windows is used)
3. Git Clone the repo
4. Change directory to the 'Deep Learning Models folder'
5. Download the pre-trained glove embedding file and put it in this folder. 
The direct link is here: http://nlp.stanford.edu/data/glove.6B.zip
Extract the zip folder and copy the file "glove.6B.300d.txt" into the 'Deep Learning Models" folder.
7. Back to the terminal, ensure you are in the  'Deep Learning Models" folder, if not change directory to it.
8.  You can preprocess the data (done already) by typing 'python  Data_Processing.py'
9. Now that the data is cleaned, you can model it with existing implementations of a basic CNN and RNN models by typing "python Modeling.py"
10. The accuracy results of the results of these models is appended to the text file "DL-Model_Accuracy.txt". Open it and scroll down to the last appended block of text.

# Acknowledgements:
We would like to credit Dr. Wei Le and classmates in COMS 665 (Spring 2020) - Iowa State University for advising us and give comments to improve the project. We also want to acknowledge authors of TSE 2018 paper (IEEE TSE2018: A deep learning model for estimating story points) for releasing the dataset.
