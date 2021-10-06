#ifndef __PYRAMID_H__
#define __PYRANID_H__

#include <vector>
#include <opencv.hpp>

using namespace cv;
using namespace std;

static unsigned int calc_max_pyr_layers(int img_w, int img_h);
unsigned int calc_pyr_layers(int img_w, int img_h, int input_layer_num);
int create_pyramid(Mat* p_src, vector<Mat> *p_gauss_pyr, vector<Mat> *p_laplace_pyr,
	int layer_num = -1, int pyr_type = CV_16SC3);
int create_gauss_pyramid(Mat* p_src, vector<Mat> *p_gauss_pyr,
	int layer_num = -1, int pyr_type = CV_16SC3);
int create_laplace_pyramid_by_gauss(vector<Mat> *p_laplace_pyr, vector<Mat> *p_gauss_pyr);
int create_laplace_pyramid(Mat* p_src, vector<Mat> *p_laplace_pyr, 
	int layer_num = -1, int pyr_type = CV_16SC3);
int reconstruct_pyrmaid(vector<Mat> *p_laplace_pyr, Mat* p_dst);
#endif // !__PYRAMID__H__
