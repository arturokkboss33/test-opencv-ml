test-opencv-ml
==============

C++ code simple examples to use the OpenCV ML library, so far:
- decision trees (both opencv and the author's implementations)
- random forests (opencv)

description
===========

For testing, the dataset in http://archive.ics.uci.edu/ml/datasets/SPECT+Heart was used, as well as a simple self-made example.

The accuracy achieved with the ML methods applied so far are:

- Mehtod applied by the database's authors (CLIP3) -- 84%
- Decision trees (author's implementation) --- 79.7%
- Decision trees (opencv) --- 80.3%
- Random forests (opencv) --- 82.9%

comments
========

The previous results were achieved by modifying manually the paramertes of each ML method. Better accuracy in the data set can be achieve 
by compromising the accuracy in the tranining set.

installation
============

You only need to install g++ compiler.

how to
======

1) For the author's decision tree implementation (my_dectree folder):

- Download the files. 

- Run make. 

- Run the exe file generated with the next instruction:

./dectree_exe FILE

Where FILE is the base name of your training and test dataset (without extension).

Name your training dataset as FILE.train and your test dataset as FILE.test, and see the examples in the project to ensure their correct parsing. 
(And save the files in the correct folder!)

The first column in the files is the output classification, and the rest is the binary value of the cases' attributes.

2) For the rest ML methods (opencv_dectree):

- Download the files.

- In the command line write: ./compile_tree_ex.sh FILE_NAME.cpp ; where FILE_NAME is the C++ file you want to compile.

- Run the exe file generated as: ./FILE_NAME FILE (this instruction has been explained)
