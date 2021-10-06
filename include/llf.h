#ifndef __LLF_H__
#define __LLF_H__

#include <vector>
#include <opencv.hpp>

using namespace cv;
using namespace std;

Mat remap_mat(Mat *p_src, double* g_gauss);
Mat remap_mat2(Mat *p_src, double* g_gauss);
int llf_orig_NN(Mat* p_src, Mat* p_dst, int layer_num = -1, int pyr_type = CV_16SC3);

int llf_fast_0(Mat* p_src, Mat* p_dst, int N = 16, int layer_num = -1, int pyr_type = CV_16SC3);
int llf_fast_1(Mat* p_src, Mat* p_dst, int N = 16, int layer_num = -1, int pyr_type = CV_16SC3);
double remap_func(double g0, double gi);
#endif 
