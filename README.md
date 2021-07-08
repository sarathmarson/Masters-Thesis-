# Master's Thesis:- Effects of Transfer Learning on Chess Piece Recognition

This repository contains detailed overview of my master's thesis- 'Effects of Transfer Learning on Chess Piece Recognition'. 


## Steps 
Following are the steps required to perform chess piece recognition using my algorithm.

### Clone reference repository

The steps for performing detection using YOLOv3 is described in detail in https://github.com/zzh8829/yolov3-tf2 . This repsoitory is cloned in to our colab working environemnt using the following operation: 

    !git clone https://github.com/zzh8829/yolov3-tf2.git

### Clone the thesis repository into reference repository

Then, the python scripts given in this repository are cloned into the above repository. For cloning this repsoitory into colab, use the command shown below:

    !git clone https://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@github.com/sarathmarson/Masters-Thesis-.git
    
### Weights of datasets

The trained weights of three different datasets can be downloaded from the links given below:

First chessboard data: https://drive.google.com/drive/folders/1aqGtr6g33wydPfuyJZREFyp73YHcrW3N?usp=sharing

Second chessboard data: https://drive.google.com/drive/folders/1OvPxm3VnZXlhcpKXZjSb9Vn5Anttiq1Z?usp=sharing

Mixed dataset: https://drive.google.com/drive/folders/1gsqf2lcHBDezKyWK3WORbgXg9UxSNNbL?usp=sharing

### Output obtained for each case using the detection algorithm

Case 1: First Chessboard using weights of first chessboard data
A python script for detection is run on the PC with weights for the first chessboard data. The twelve different classes are detected by the system, and their
names are displayed in bounding boxes along with their confidence of prediction as shown in figure:

![alt text](images/ch_sm_ori.png)

Case 2: Second Chessboard using weights of second chessboard data
The python script for detecting pieces of the second chessboard is executed using the corresponding weights. The output obtained for the second chessboard along with prediction confidence is shown in the following figure:

![alt text](images/ch_big_ori.png)

Case 3: First Chessboard using weights of mixed data
Testing of first chess board is performed using the mixed weights of both first and second chessboard data. A sample of detection obtained along with the percentage of confidence is illustrated in the figure below: 

![alt text](images/ch_sm_mix.png)

Case 4: Second Chessboard using weights of mixed data
Mixed weights obtained by training both first and second chessboard data are now used to test the detection in the second chessboard. The figure below depicts the results obtained:

![alt text](images/ch_big_mix.png)
















