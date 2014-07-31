/*
 * =====================================================================================
 *
 *       Filename:  opencv_dectree_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/30/2014 02:49:25 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <fstream>


//std::vector< std::vector<float> > training_set;
cv::Mat training_set;
cv::Mat responses;


void load_trainset(std::string filename, cv::Mat& datai)
{

	//Variables for parsing the data file
	std::string line;
	std::stringstream parse;
	int ssize = 100; //establish a buffer size to store attribute values,
			 //which for binary classification string are no bigger than 1
	char c[ssize];
	char delimiter = ',';

	//Variables to store the values in the data file
	//std::vector<float> tmpcase;

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
				if(c_atr == 0)
					responses.push_back((float)atof(c));
				else
					tmpcase.push_back((float)atof(c));

				c_atr++;
			}

			parse.str(""); //safety measure to erase previous contents
			parse.clear(); //clear flags to be able to read from it again
			
			tmpcase = tmpcase.t();
			training_set.push_back(tmpcase);
			~tmpcase; 
		}
	}

}


std::vector<double> test_tree(std::string filename, CvDTree* dectree)
{
	std::vector<double> results;
	cv::Mat mask(1,training_set.cols,CV_8U, cv::Scalar(0));
	//Variables for parsing the data file
	std::string line;
	std::stringstream parse;
	int ssize = 100; //establish a buffer size to store attribute values,
			 //which for binary classification string are no bigger than 1
	char c[ssize];
	char delimiter = ',';

	//Variables to store the values in the data file
	//std::vector<float> tmpcase;

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
			std::cout << tmpcase << std::endl;
			double r = dectree->predict(tmpcase,mask)->value;
			std::cout << r << " * " << tmpresult << std::endl;
			results.push_back(r);
			int d = fabs(r-tmpresult) >= FLT_EPSILON;
			std::cout << d << std::endl;
			if(d)
			{
				mis_classif++;
				std::cout << "+++" << std::endl;
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

int
main ( int argc, char *argv[] )
{
	//Read database
	cv::Mat training_data;
	//variables to parse the console input and search te trainig and test files
	std::string train_dir = "../TrainSets/";
	std::string test_dir = "../TestSets/";
	std::string train_ext = ".train";
	std::string test_ext = ".test";
	std::string train_file = train_dir+argv[1]+train_ext;	
	std::string test_file = test_dir+argv[1]+test_ext;	
	load_trainset(train_file, training_data);
	std::cout << responses.row(0) << std::endl;
	std::cout << training_set.row(0) << std::endl;
	std::cout << training_set.type() << std::endl;

	 
	//Create tree
	CvDTree* dtree;
	//CvMat* var_type;
	//CvMat* missing;
	float priors[] = {1.,1.};

	//var_type = cvCreateMat(training_set.cols+1,1,CV_8U);//specify if the attributes and responses are ordered or categorical
	//cvSet(var_type, cvScalarAll(CV_VAR_CATEGORICAL)); //set the matrix 
	cv::Mat missing(training_set.size(),CV_8U,cv::Scalar(0));
	cv::Mat var_type(training_set.cols+1,1,CV_8U, cvScalarAll(CV_VAR_CATEGORICAL));	


	dtree = new CvDTree;

	dtree->train(training_set, CV_ROW_SAMPLE, responses, cv::Mat(), cv::Mat(), var_type, missing, 
		     CvDTreeParams(300, //max_depth
		     		   0,  //min_sample_count
				   0,  //regression accuracy, N/A for categorical
				   false, //compute surrogate split
				   300, //max number of categories
				   1, //the number of cross-validation folds to prune
				   false, //harsher pruning
				   true, //throw away pruned branches
				   priors //array of priors (weights)
				   ));	
	std::cout << "Training ready..." << std::endl;	

	std::cout << "Testing cases..." << std::endl;
	std::vector<double> predictions;
	predictions = test_tree(test_file,dtree);

	cv::Mat imp = dtree->getVarImportance();
	std::cout << imp << std::endl;


	return 0;
}				/* ----------  end of function main  ---------- */
