This is an implementation of ECIR 2019 paper :  
Unsupervised Explainable Controversy Detection from Online News : [Paper Link](http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1334)

#### Data Downloads

Download the following link and unpack it to the data directory.

[Data Download](https://drive.google.com/file/d/1g6yh77tBgWlgXcKCLULLBVcUt7Xvs1JE/view?usp=sharing)



#### Requirements

* Python 3.6
* tensorflow=1.8.0


#### Input Data

* Most data are stored in pickle format(v3)
* All input data for article and comments are already converted into array of term indices.
  * code_articles.pickle : List of articles
  * code_comments.pickle : List of comments. 
  * For structure of each pickle's content, read the functions in load_data.py  
* voca2idx.pickle contains the term to index conversion information
  * Using it, you can write code to recover original texts of the articles and comments.  

#### Training

* main.py executes the training steps for Language Model based controversy classfier.
* By default it loads pre-trained disagreement classifier


#### Topic Extraction

* explain.py executes the topic extractions
* Using the trained controversy classifier, it extracts the controversial topic

#### (Dis)Agreement training

* Disagreement expression detector.
* We preprocessed AAWD (Authority and Alignment in Wikipedia Discussions) data into commentAgree.txt file.  
* commentAgree.txt contains 3-way classification training data where 
  * 0 indicates none.
  * 1 indicates agreement expression, 
  * 2 indicates disagreement expression
  
* Training data orignally from the paper : Annotating social acts: Authority claims and
alignment moves in wikipedia talk pages.

* Model 
  * Convolutional Neural Network based (cnn.py)


#### Contact


contact : youngwookim@cs.umass.edu

