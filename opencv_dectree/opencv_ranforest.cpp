/* =============================================================*/
/* --- RANDOM FOREST - OPENCV DECISION TREE CLASS            ---*/
/* FILENAME: opencv_ranforest.cpp 
 *
 * DESCRIPTION: basic example on how to use decision trees in 
 * opencv
 *
 * VERSION: 1.0
 *
 * CREATED: 07/30/2013
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <fstream>

//Method to parse a file containing training data for a decision tree
void load_trainset(std::string filename, cv::Mat& data, cv::Mat& resp)
{

	//Variables for parsing the data file
	std::string line;
	std::stringstream parse;
	int ssize = 100; //establish a buffer size to store attribute values,
			 //which for binary classification string are no bigger than 1
	char c[ssize];
	char delimiter = ',';


	std::ifstream dataset_file(filename.c_str(), std::ios::in);

	if(!dataset_file)
	{
		std::cerr << "Cannot load training set file" << std::endl;
	}
	else
	{
		while( (getline(dataset_file, line))!= NULL )
		{
			parse << line;
			cv::Mat tmpcase;
			int c_atr = 0;

			while( parse.getline(c,ssize,delimiter) )
			{
				if(c_atr == 0) //the 1st column contains the response
					resp.push_back((float)atof(c));
				else
					tmpcase.push_back((float)atof(c));

				c_atr++;
			}

			parse.str(""); //safety measure to erase previous contents
			parse.clear(); //clear flags to be able to read from it again
			
			tmpcase = tmpcase.t();
			data.push_back(tmpcase);
			~tmpcase; 
		}
	}

}


std::vector<double> test_ranforest(std::string filename, CvRTrees* forest, int no_attributes)
{
	std::vector<double> results;
	cv::Mat mask(1,no_attributes,CV_8U, cv::Scalar(0));
	//Variables for parsing the data file
	std::string line;
	std::stringstream parse;
	int ssize = 100; //establish a buffer size to store attribute values,
			 //which for binary classification string are no bigger than 1
	char c[ssize];
	char delimiter = ',';

	std::ifstream testset_file(filename.c_str(), std::ios::in);

	if(!testset_file)
	{
		std::cerr << "Cannot load test set file" << std::endl;
	}
	else
	{
		int no_cases = 0;
		int mis_classif = 0;

		while( (getline(testset_file, line))!= NULL )
		{
			parse << line;
			cv::Mat tmpcase;
			double tmpresult;
			int c_atr = 0;

			while( parse.getline(c,ssize,delimiter) )
			{
				if(c_atr == 0)
					tmpresult = atof(c);
				else
					tmpcase.push_back((float)atof(c));

				c_atr++;
			}

			parse.str(""); //safety measure to erase previous contents
			parse.clear(); //clear flags to be able to read from it again
			
			tmpcase = tmpcase.t();
			double pred_result = forest->predict(tmpcase,mask); //obtain prediction value
			results.push_back(pred_result);
			int isPredCorrect = fabs(pred_result-tmpresult) >= FLT_EPSILON; //check if the prediction is accurate
			
			//for debugging
			/* 
			std::cout << tmpcase << std::endl;
			std::cout << pred_result << " * " << tmpresult << std::endl;
			std::cout << isPredCorrect << std::endl;
			*/

			if(isPredCorrect)
			{
				mis_classif++;
				//std::cout << "+++" << std::endl;
			}
			no_cases++;
			~tmpcase; 
		}

		double error = (double)mis_classif/no_cases;
		std::cout << "Wrong class: " << mis_classif << " No cases: " << no_cases;
		std::cout << std::endl;
		std::cout << "Error %: " << error << std::endl;
	}

	return results;


}


//MAIN METHOD
int
main ( int argc, char *argv[] )
{
	//Variables to store information about the training set
	cv::Mat attr_mat;
	cv::Mat resp_vec;
	
	//variables to parse the console input and search te trainig and test files
	std::string train_dir = "../TrainSets/";
	std::string test_dir = "../TestSets/";
	std::string train_ext = ".train";
	std::string test_ext = ".test";
	std::string train_file = train_dir+argv[1]+train_ext;	
	std::string test_file = test_dir+argv[1]+test_ext;

	//fill the matrixes with the training data from a file
	//here the CvMLData class can be used if the information is in .csv format	
	load_trainset(train_file, attr_mat, resp_vec);
	//for debugging
	/*
	std::cout << resp_vec.row(0) << std::endl;
	std::cout << attr_mat.row(0) << std::endl;
	std::cout << attr_mat.type() << std::endl;
	*/
	
	//--- CREATE RANDOM FOREST ---//
	CvRTrees* forest;
	cv::Mat var_type(attr_mat.cols+1,1,CV_8U, cvScalarAll(CV_VAR_CATEGORICAL));
	cv::Mat missing(attr_mat.size(), CV_8U, cv::Scalar(0));
	float priors[] = {1.,1.};
	
	std::cout << "Training random forest..." << std::endl;
	forest = new CvRTrees;
	forest->train(	attr_mat,	//cv::Mat containing samples and their attribute values
			CV_ROW_SAMPLE,	//defines if there is a sample ine very row or col
			resp_vec,	//vector containing the responses of every sample
			cv::Mat(),	//vector to indicate which attributes to consider for the training (0-skip)
			cv::Mat(),	//vector to indicate which samples to consider for the training (0-skip)
			var_type,	//matrix that states if eacg feature is ordered or categorical
			missing,	//matrix used to indicate missing values with a 1
			CvRTParams(	10, 	//max depth of the tree
					10,	//min number of samples in a node to make a split
					0,	//regression acuracy, N/A for categorical, termination criteria for regression
					true,  //compute surrogate splits
					100, 	//max number of categories
					priors,	//array of priors (weights)
					true,	//calculate var importance
					0,	//active vars, number of variables used to build each tree node
					100,	//max number of trees in the forest
					0.01,  //sufficient accuracy (OOB error)
					CV_TERMCRIT_ITER //termination criteria, by reaching max number of trees and/or accuracy
			));

	std::cout << "Ready" << std::endl;

	//--- RANDOM FOREST PREDICTION ---//
	std::cout << "No trees: " << forest->get_tree_count() << std::endl;
	std::cout << "Calculating training error..." << std::endl;
	std::cout << "Error %: " << forest->get_train_error() << std::endl;

	std::cout << "Evaluating test cases..." << std::endl;
	std::vector<double> predictions;
	predictions = test_ranforest(test_file,forest,attr_mat.cols);

	//--- OTHER RF METHODS ---//
	//variable importance
	cv::Mat var_imp = forest->getVarImportance();
	std::cout << "Variable importance: " << std::endl;
	std::cout << var_imp << std::endl;
	//proximity
	//std::cout << attr_mat.rows << std::endl;
	cv::Mat sample1 = attr_mat.row(11).clone();		
	cv::Mat sample2 = attr_mat.row(12).clone();
	//std::cout << sample1 << std::endl;
	CvMat s1 = sample1;
	CvMat s2 = sample2;
	std::cout << "Proximities" << std::endl;
	std::cout << forest->get_proximity(&s1,&s2) << std::endl;		

 
	return 0;
}				/* ----------  end of function main  ---------- */
