#include "include/pyramid.h"

static unsigned int calc_max_pyr_layers(int img_w, int img_h)
{
	unsigned int max_layer_num = 0;
	//max_layer_num = ceil(log(min(img_w, img_w)) - log(2.0)) + 2;
	max_layer_num = ceil(log(min(img_w, img_h)) / log(2.0)) + 1;
	return max_layer_num;
}

unsigned int calc_pyr_layers(int img_w, int img_h, int input_layer_num)
{
	unsigned int max_layer_num = calc_max_pyr_layers(img_w, img_h);
	if (input_layer_num < 0){
		return max_layer_num;
	}
	// pyr layers should >= 2
	if (input_layer_num < 2){
		return 2;
	}

	return MIN(input_layer_num, max_layer_num);
}

int create_pyramid(Mat* p_src, vector<Mat> *p_gauss_pyr, vector<Mat> *p_laplace_pyr, int layer_num, int pyr_type)
{
	// 
	if (p_gauss_pyr == NULL && p_laplace_pyr == NULL){
		return -1;
	}
	if (p_gauss_pyr != NULL && p_laplace_pyr == NULL){
		return create_gauss_pyramid(p_src, p_gauss_pyr, layer_num, pyr_type);
	}
	if (p_laplace_pyr != NULL && p_gauss_pyr == NULL){
		return create_laplace_pyramid(p_src, p_laplace_pyr, layer_num, pyr_type);
	}

	int layer_idx = 0;
	Mat mat_up, mat_down, mat_diff, mat_gauss;

	layer_num = calc_pyr_layers(p_src->rows, p_src->cols, layer_num);

	// 类型最好为有符号类型
	p_src->convertTo(mat_gauss, pyr_type);

	p_gauss_pyr->clear();
	p_gauss_pyr->assign(layer_num, Mat());
	p_laplace_pyr->clear();
	p_laplace_pyr->assign(layer_num, Mat());


	for (layer_idx = 0; layer_idx < layer_num - 1; layer_idx++)
	{
		(*p_gauss_pyr)[layer_idx] = mat_gauss.clone();

		// down
		cv::pyrDown(mat_gauss, mat_down);

		// up
		cv::pyrUp(mat_down, mat_up, mat_gauss.size());

		// diff
		mat_diff = mat_gauss - mat_up;
		(*p_laplace_pyr)[layer_idx] = mat_diff.clone();

		mat_gauss = mat_down;
	}

	(*p_gauss_pyr)[layer_num - 1] = mat_gauss.clone();
	(*p_laplace_pyr)[layer_num - 1] = mat_gauss.clone();

	return 0;
}


int create_gauss_pyramid(Mat* p_src, vector<Mat> *p_gauss_pyr, int layer_num, int pyr_type)
{
	int layer_idx = 0;
	Mat mat_up, mat_down;
    
	layer_num = calc_pyr_layers(p_src->rows, p_src->cols, layer_num);

	// 类型最好为有符号类型
	p_src->convertTo(mat_up, pyr_type);

	// 
	p_gauss_pyr->clear();
	p_gauss_pyr->assign(layer_num, Mat());

	for (layer_idx = 0; layer_idx < layer_num - 1; layer_idx++)
	{
		(*p_gauss_pyr)[layer_idx] = mat_up.clone();
		cv::pyrDown(mat_up, mat_down);
		mat_up = mat_down;
	}

	(*p_gauss_pyr)[layer_num - 1] = mat_up.clone();
	return 0;
}


int create_laplace_pyramid_by_gauss(vector<Mat> *p_laplace_pyr, vector<Mat> *p_gauss_pyr)
{
	int layer_num = p_gauss_pyr->size();
	int layer_idx = 0;
	Mat mat_up, mat_down;
	Mat mat_gauss, mat_diff;
	Size size_up;

	p_laplace_pyr->clear();
	p_laplace_pyr->assign(layer_num, Mat());

	// layer max - 1
	mat_down = (*p_gauss_pyr)[layer_num - 1];
	(*p_laplace_pyr)[layer_num - 1] = mat_down.clone();

	for (layer_idx = layer_num - 2; layer_idx >=0 ; layer_idx--)
	{
		// layer up
		mat_gauss = (*p_gauss_pyr)[layer_idx];
		size_up = mat_gauss.size();

		// diff
		cv::pyrUp(mat_down, mat_up, size_up);
		mat_diff = mat_gauss - mat_up;
		(*p_laplace_pyr)[layer_idx] = mat_diff.clone();
		mat_down = mat_gauss;
	}

	return 0;
}


int create_laplace_pyramid(Mat* p_src, vector<Mat> *p_laplace_pyr, int layer_num, int pyr_type)
{
	int layer_idx = 0;
	Mat mat_up, mat_down;
	Mat mat_gauss, mat_diff;
	Size size_up;

	layer_num = calc_pyr_layers(p_src->rows, p_src->cols, layer_num);

	p_laplace_pyr->clear();
	p_laplace_pyr->assign(layer_num, Mat());

	// layer max - 1
	p_src->convertTo(mat_gauss, pyr_type);

	for (layer_idx = 0; layer_idx < layer_num - 1; layer_idx++)
	{
		// layer down
		cv::pyrDown(mat_gauss, mat_down);

		// layer up
		size_up = mat_gauss.size();
		cv::pyrUp(mat_down, mat_up, size_up);

		// diff
		mat_diff = mat_gauss - mat_up;
		(*p_laplace_pyr)[layer_idx] = mat_diff.clone();

		mat_gauss = mat_down;
	}

	(*p_laplace_pyr)[layer_num - 1] = mat_gauss.clone();

	return 0;
}


int reconstruct_pyrmaid(vector<Mat> *p_laplace_pyr, Mat* p_dst)
{
	int layer_num = p_laplace_pyr->size();
	int layer_idx = 0;
	Mat mat_up, mat_down;
	Mat mat_laplace_up;
	Size size_up;

	// layer max - 1
	mat_down = (*p_laplace_pyr)[layer_num - 1];			// 最高层laplace与高斯是相同的

	for (layer_idx = layer_num - 2; layer_idx >= 0; layer_idx--)
	{
		// layer up
		mat_laplace_up = (*p_laplace_pyr)[layer_idx];
		size_up = mat_laplace_up.size();

		// diff
		cv::pyrUp(mat_down, mat_up, size_up);
		mat_laplace_up += mat_up;

		mat_down = mat_laplace_up;
	}

	*p_dst = (*p_laplace_pyr)[0].clone();

	return 0;
}