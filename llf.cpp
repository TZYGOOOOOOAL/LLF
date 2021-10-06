#include "include/pyramid.h"
#include "include/llf.h"

// 最原始算法，时间按复杂度O(N^2)
int llf_orig_NN(Mat* p_src, Mat* p_dst, int layer_num, int pyr_type)
{
	vector<Mat> gauss_pyr;
	vector<Mat> laplace_pyr;
	vector<Mat> laplace_pyr_tmp;
	int layer_idx = 0;
	int r, c;
	double g_gauss[3];
	unsigned int data_idx = 0;

	// create gauss and laplace pyramid
	create_pyramid(p_src, &gauss_pyr, &laplace_pyr, layer_num, pyr_type);
	Mat src_mat = gauss_pyr[0];

	layer_num = gauss_pyr.size();
	short* p_gauss_data = NULL;
	short* p_laplace_data = NULL;
	short* p_laplace_tmp_data = NULL;
	Mat remap_img;
	int h, w;

	// each layer
	for (layer_idx = 0; layer_idx < layer_num - 1; layer_idx++)
	{
		printf("Calculate Layer %d\n", layer_idx);
		p_gauss_data = (short*)(gauss_pyr[layer_idx].data);
		p_laplace_data = (short*)(laplace_pyr[layer_idx].data);
		h = laplace_pyr[layer_idx].rows;
		w = laplace_pyr[layer_idx].cols;
		
		data_idx = 0;
		for (r = 0; r < laplace_pyr[layer_idx].rows; r++)
		{
			for (c = 0; c < laplace_pyr[layer_idx].cols; c++)
			{
				g_gauss[0] = p_gauss_data[data_idx];
				g_gauss[1] = p_gauss_data[data_idx + 1];
				g_gauss[2] = p_gauss_data[data_idx + 2];

				// remap
				remap_img = remap_mat(&src_mat, g_gauss);

				// laplace pyr
				create_laplace_pyramid(&remap_img, &laplace_pyr_tmp, layer_idx + 2);

				// assign
				p_laplace_tmp_data = (short*)(laplace_pyr_tmp[layer_idx].data);
				p_laplace_data[data_idx] = p_laplace_tmp_data[data_idx];
				p_laplace_data[data_idx + 1] = p_laplace_tmp_data[data_idx + 1];
				p_laplace_data[data_idx + 2] = p_laplace_tmp_data[data_idx + 2];


				data_idx += 3;

			}
		}
	}

	reconstruct_pyrmaid(&laplace_pyr, p_dst);
	p_dst->convertTo(*p_dst, p_src->type());

	return 0;
}


// fast 算法
int llf_fast_0(Mat* p_src, Mat* p_dst, int N, int layer_num, int pyr_type)
{
	// 原始影像
	Mat src_s16_img;
	p_src->convertTo(src_s16_img, pyr_type);

	// 金字塔
	vector<Mat> laplace_pyr;
	vector<Mat> gauss_pyr;
	vector<Mat> laplace_pyr_tmp;
	vector<vector<Mat>> laplace_pyr_list(N, vector<Mat>());
	
	// 循环变量
	int layer_idx = 0;
	int r, c;
	unsigned int data_idx = 0;
	int pyr_idx = 0;
	int ch_idx = 0;

	// 采样步长
	N = MAX(2, MIN(256, N));
	vector<double> steps(N, 0.0);
	double step_len = 255.0 / (N - 1);
	double step_val = 0.0;

	// init orig laplace pyr
	create_pyramid(&src_s16_img, &gauss_pyr, &laplace_pyr, layer_num, pyr_type);

	// init N laplace pyr
	for (pyr_idx = 0; pyr_idx < N; pyr_idx++)
	{
		steps[pyr_idx] = step_val;
		double g_gauss[3] = { step_val, step_val, step_val};

		Mat r_img = remap_mat(&src_s16_img, g_gauss);

		create_laplace_pyramid(&r_img, &laplace_pyr_tmp, layer_num, pyr_type);

		laplace_pyr_list[pyr_idx] = laplace_pyr_tmp;

		step_val += step_len;
	}

	// 插值
	double step_min, step_max;
	double a;
	short* p_laplace_data, *p_gauss_data;
	short* p_laplace_min_data, *p_laplace_max_data;

	layer_num = laplace_pyr_list[0].size();
	for (pyr_idx = 0; pyr_idx < N - 1; pyr_idx++)
	{
		step_min = steps[pyr_idx];
		step_max = steps[pyr_idx + 1];

		for (layer_idx = 0; layer_idx < layer_num - 1; layer_idx++)
		{
			p_gauss_data = (short*)(gauss_pyr[layer_idx].data);
			p_laplace_data = (short*)(laplace_pyr[layer_idx].data);
			p_laplace_min_data = (short*)(laplace_pyr_list[pyr_idx][layer_idx].data);
			p_laplace_max_data = (short*)(laplace_pyr_list[pyr_idx + 1][layer_idx].data);
			data_idx = 0;

			for (r = 0; r < gauss_pyr[layer_idx].rows; r++)
			{
				for (c = 0; c < gauss_pyr[layer_idx].cols; c++)
				{
					for (ch_idx = 0; ch_idx < 3; ch_idx ++)
					{
						a = p_gauss_data[data_idx] - step_min;
						if (a < step_len && a >= 0)		// 必须a>0, 防止重复累加
						{
							a /= step_len;
							p_laplace_data[data_idx] = p_laplace_min_data[data_idx] * (1 - a) + \
								p_laplace_max_data[data_idx] * a;
						}
						data_idx++;
					}
				}
			}// end one layer
		}// end one pyr
	}

	reconstruct_pyrmaid(&laplace_pyr, p_dst);
	p_dst->convertTo(*p_dst, p_src->type());

	return 0;
}


// fast 算法，减少内存占用
int llf_fast_1(Mat* p_src, Mat* p_dst, int N, int layer_num, int pyr_type)
{
	// 原始影像
	Mat src_s16_img;
	p_src->convertTo(src_s16_img, pyr_type);

	// 金字塔
	vector<Mat> laplace_pyr;
	vector<Mat> gauss_pyr;
	vector<Mat> laplace_pyr_tmp;

	// 循环变量
	int layer_idx = 0;
	int r, c;
	unsigned int data_idx = 0;
	int pyr_idx = 0;
	int ch_idx = 0;

	// 采样步长
	N = MAX(2, MIN(256, N));
	double step_len = 255.0 / (N - 1);
	double step_val = 0.0;

	// init orig laplace pyr
	create_pyramid(&src_s16_img, &gauss_pyr, &laplace_pyr, layer_num, pyr_type);
	layer_num = gauss_pyr.size();

	// 插值
	double a;
	short* p_laplace_data, *p_gauss_data;
	short* p_laplace_tmp_data;

	// init N laplace pyr
	for (pyr_idx = 0; pyr_idx < N; pyr_idx++)
	{
		double g_gauss[3] = { step_val, step_val, step_val };

		Mat r_img = remap_mat2(&src_s16_img, g_gauss);

		create_laplace_pyramid(&r_img, &laplace_pyr_tmp, layer_num, pyr_type);

		for (layer_idx = 0; layer_idx < layer_num - 1; layer_idx++)
		{
			p_gauss_data = (short*)(gauss_pyr[layer_idx].data);
			p_laplace_data = (short*)(laplace_pyr[layer_idx].data);
			p_laplace_tmp_data = (short*)(laplace_pyr_tmp[layer_idx].data);
			data_idx = 0;

			for (r = 0; r < gauss_pyr[layer_idx].rows; r++)
			{
				for (c = 0; c < gauss_pyr[layer_idx].cols; c++)
				{
					for (ch_idx = 0; ch_idx < 3; ch_idx++)
					{
						a = fabs(p_gauss_data[data_idx] - step_val);
						if (a < step_len){
							p_laplace_data[data_idx] += p_laplace_tmp_data[data_idx] * (1 - a / step_len);
						}
						data_idx++;
					}
				}
			}// end one layer
		}// end one pyr

		step_val += step_len;
	}

	reconstruct_pyrmaid(&laplace_pyr, p_dst);
	p_dst->convertTo(*p_dst, p_src->type());

	return 0;
}


double remap_func(double g_img, double g_gauss)
{
	double result = 0.0;
	double sigma = 0.3;
	double alpha = 15;
	double beta = 1.0;
	g_gauss /= 255.0;
	g_img /= 255.0;
	double delta = g_img - g_gauss;
	double delta_abs = fabs(delta);
	int sign = (delta >= 0) ? 1 : -1;

	if (fabs(delta) <= sigma){
		result = g_gauss + sign * sigma * 
			std::pow((delta_abs / sigma), alpha);
	}
	else{
		result = g_gauss + 
			sign * (beta * (delta_abs -sigma) + sigma);
	}

	result = MAX(0.0, MIN(1.0, result));
	result *= 255;
	return result;
}

Mat remap_mat(Mat *p_src, double* g_gauss)
{
	int width = p_src->cols;
	int height = p_src->rows;
	int r, c;
	int g0;
	Mat remap_img(p_src->size(), p_src->type());

	// TODO 
	assert(p_src->isContinuous() && remap_img.isContinuous());
	unsigned int data_idx = 0;
	short* p_src_data = (short*)(p_src->data);
	short* p_dst_data = (short*)(remap_img.data);


	for (r = 0; r < height; r++)
	{
		for (c = 0; c < width; c++)
		{
			// B
			g0 = p_src_data[data_idx];
			p_dst_data[data_idx] = remap_func(g0, g_gauss[0]);
			// G
			g0 = p_src_data[data_idx + 1];
			p_dst_data[data_idx + 1] = remap_func(g0, g_gauss[1]);
			// R
			g0 = p_src_data[data_idx + 2];
			p_dst_data[data_idx + 2] = remap_func(g0, g_gauss[2]);

			data_idx += 3;
		}
	}
	
	return remap_img;
}


double remap_func2(double g_img, double g_gauss)
{
	double result = 0.0;
	double sigma = 0.15;
	double fact = 2;

	g_gauss /= 255.0;
	g_img /= 255.0;
	double delta = g_img - g_gauss;

	result = fact * delta * exp(-delta * delta / (2 * sigma*sigma));
	
	result = MAX(0.0, MIN(1.0, result));
	result *= 255;
	return result;
}

Mat remap_mat2(Mat *p_src, double* g_gauss)
{
	int width = p_src->cols;
	int height = p_src->rows;
	int r, c;
	int g0;
	Mat remap_img(p_src->size(), p_src->type());

	// TODO 
	assert(p_src->isContinuous() && remap_img.isContinuous());
	unsigned int data_idx = 0;
	short* p_src_data = (short*)(p_src->data);
	short* p_dst_data = (short*)(remap_img.data);


	for (r = 0; r < height; r++)
	{
		for (c = 0; c < width; c++)
		{
			// B
			g0 = p_src_data[data_idx];
			p_dst_data[data_idx] = remap_func2(g0, g_gauss[0]);
			// G
			g0 = p_src_data[data_idx + 1];
			p_dst_data[data_idx + 1] = remap_func2(g0, g_gauss[1]);
			// R
			g0 = p_src_data[data_idx + 2];
			p_dst_data[data_idx + 2] = remap_func2(g0, g_gauss[2]);

			data_idx += 3;
		}
	}

	return remap_img;
}