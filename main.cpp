#include <stdio.h>
#include <string>
#include "include/pyramid.h"
#include "include/llf.h"

int main()
{
	string img_path = "C:/Users/tzy/Desktop/test/sg.png";
	Mat img = imread(img_path);
	Mat reconstruct_img;

	//cv::resize(img, img, Size(300, 400));
	Mat result(img.size(), img.type());

	vector<Mat> gauss_pyr;
	vector<Mat> laplace_pyr;

	//create_pyramid(&img, &gauss_pyr, &laplace_pyr, 1);
	//reconstruct_pyrmaid(&laplace_pyr, &reconstruct_img);
	//reconstruct_img.convertTo(result, CV_8UC3);

	//llf_orig_NN(&img, &result);
	llf_fast_1(&img, &result, 16);

	imshow("result", result);
	imshow("result_src", img);
	cv::waitKey(0);
	return 0;
}