# Software Effort Prediction in Agile Development with Machine Learning & Natural Language Processing
The success of software development projects hinges upon, amongst other factors, on project and time management. One popular method used to aid time management and estimating project timelines for agile software development is the estimation number of story points, which represent the amount of development effort per individual software issues or requests in number of man hours. In this paper, we explore various text vectorization machine learning techniques to predict software development effort measured in number of story points. Our results show that the problem can be formulated as both a classification or a regression problem, and successfully solved using supervised learning. Moreover, several of our regression models achieve better accuracy than prior literature. We also demonstrate that Generative Adversarial Network (GAN) semi-supervised yields significantly better results than normal semi-supervised learning. Finally, we show that deep learning architectures such as convolutional attention recurrent neural networks yield very promising results, but require further hyperparameter tuning.

**Authors: Hung Phan, Eliska Koberdanz, Jeremiah Roghair** from Department of Computer Science, Iowa State University, USA.

This is the research project in the course Advance in Software Engineering - COMS 665 (Spring 2020) - Iowa State University, USA.

# Requirements:
- Python 3.6.
- Scikit-learn 0.22.2.
- Keras 2.2.5.
- PyTorch 1.5.
- TensorFlow 2.0.

# Instruction for Replication:

We put all of our code, data and result inside replicationPackage folder. This folder has following items:

- preprocessCode: the code for vectorizing software features of TSE 2018 dataset.
- expectedResults: expected results for Research Question 1-4 (you can generate the results by yourselves if you run the python code inside the replicationPackage folder.
- data: Including dataset From TSE 2018 paper, 50000 story text for training Doc2Vec, pretrainedVector in 5 vectorization techniques.
- RQ3: all of the code for running semi supervised learning. We inherit and update the code from this paper: https://github.com/yanlirock/RLANS
- Other python codes: code for running machine learning experiments.

We set the default of any result you generate at the 'result' folder. However, you can go to the code and change the default paths to your expected locations.

The following code will run for evaluating story points by Machine Learning algorithms on our pretrained vectors. If you want to generate the vector by yourselves, please run each files in the preprocessCode folder.

**RQ1. Classification: What is the accuracy of story points categories classification?**

- Run 'evaluateRQ1.py'.
- Input: the default is the path 'replicationPackage/data/pretrainedVector/TFIDF4/' (you can go to the code and change paths for other vectorization models)
- Output: the accuracy of each systems in 10 ML Classification algorithms. You can see the best accuracy on each systems along with the details prediction in 'details/' folder.


**RQ2. Regression: What is the accuracy of story points prediction?**

- Run 'evaluateRQ2.py'.
- Input: the default is the path 'replicationPackage/data/pretrainedVector/TFIDF4/' (you can go to the code and change paths for other vectorization models)
- Output: the accuracy of each systems in 7 ML Regression algorithms. You can see the best accuracy on each systems along with the details prediction in 'details/' folder.

**RQ3. Semi-supervised training: How does the semi-supervised training affect accuracy?**

- Go inside the RQ3 sub-folder.

First, you need to run the language model creation for unlabeled data:
- Run 'python language_model_training.py --cuda --batch_size=32 --lr=0.01 --reduce_rate=0.9 --save='/ag_lm_model/''

Next, You can run 3 following configurations:
- For supervised classification, run:
python classifier_training.py --cuda --lr=0.001 --batch_size=128 --save='/classify_no_pre/' --pre_train='' --number_per_class=1000 --reduce_rate=0.95
- For original semi supervised classification, run:
python classifier_training.py --cuda --lr=0.001 --batch_size=128 --save='/classify_with_pre/' --pre_train='/ag_lm_model' --number_per_class=1000 --reduce_rate=0.95

- For GAN semi supervised classification, run:
python Adversarial_training.py --cuda --lr=0.001 --batch_size=128 --save='/ag_adv_model/' --pre_train='/ag_lm_model' --number_per_class=1000 --reduce_rate=0.95

You will see the detail...txt as the output prediction.


**RQ4. Hyperparameter Tuning: Does hyperparameter tuning improve the accuracy?**

- For RQ 4.1, run 'evaluateRQ4-1.py'. Output: the tuning result on classification of 'talendesb' system.
- For RQ 4.2, run 'evaluateRQ4-2.py'. Output: the tuning result on regression of 'talendesb' system.

**RQ5. Vectorization: What text vectorization techniques are most suitable?**

- You can see the results after running evaluations of RQ1 and RQ2.


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
We would like to credit Dr. Wei Le and classmates in COMS 665 (Spring 2020) - Iowa State University for advise us and give comments to improve the project. We also want to acknowledge authors of TSE 2018 paper (IEEE TSE2018: A deep learning model for estimating story points) for releasing the dataset and Yan Li for publishing the code for GAN Text Classification in KDD 2018 paper (https://github.com/yanlirock/RLANS).
