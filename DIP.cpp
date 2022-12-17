// DIP.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <cmath> 
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

#define NUM_channels 3
#define MAX_level 256 
#define BRIGHTNESS 150

/*
* INPUT  -> Input image matrix
  OUTPUT -> Displays number of colomns number of rows and number of channels
  if number of bits used for one pixel is not 8 bits then it will terminates program
*/
void printD(Mat image)
{

	cout << " Number of colomns " << image.cols;
	cout << "\n Number of rows " << image.rows;
	cout << "\n Number of channels " << image.channels();
	if (image.channels() != NUM_channels)
		exit(0);

	switch (image.depth())
	{
		case CV_8U: cout << "\n Number of bits used per pixel is 8";
					break;
		default: cout << "\n Given image depth is not 8";
				 exit(0);
	}
}


/*
*  INPUT  -> Image matrix and Histogram array
*  OUTPUT -> Histogram of given image and Return intensity of image
*  intensity of image is (R+G+B)/3
*/
float** Histogram_cal(Mat IN, float* hist)
{
	float temp = 0;
	// Memory assignement for INTENSITY array
	float** IN_inten = new float* [IN.rows];
	for (int i = 0; i < IN.rows; i++)
	{
		IN_inten[i] = new float[IN.cols];
	}

	// Intialization of histogram array to zeros
	for (int i = 0; i < MAX_level; i++)
	{
		hist[i] = 0;
	}


	//if (IN.channels() == NUM_channels)
	//{
		// HISTOGRAM caluclation for given color image
		for (int i = 0; i < IN.rows; i++)
		{
			for (int j = 0; j < IN.cols; j++)
			{
				temp = 0;
				for (int k = 0; k < NUM_channels; k++)
				{
					temp = temp + (uint)IN.data[i * IN.cols * NUM_channels + j * NUM_channels + k];
				}
				temp = temp / NUM_channels;
				IN_inten[i][j] = temp;
				hist[(int)temp]++;
			}
		}

	//}
	//else
	//{
	//	// HISTOGRAM caluclation for given gray scale image
	//	for (int i = 0; i < IN.rows; i++)
	//	{
	//		for (int j = 0; j < IN.cols; j++)
	//		{
	//			IN_inten[i][j] = (uint)IN.data[i * IN.cols + j];
	//			hist[(uint)IN.data[i * IN.cols + j]]++;
	//		}
	//	}
	//}
	return IN_inten;
}


/*
* INPUT  -> Histogram array of 256 size: Name contains name of the histogram to be displayed
* OUTPUT -> using imshow histogram graph image is displayed
* to change the color of the histogram graph change value of BRIGHTNESS
*/
void HistogramDisplay(float* hist, string Name)
{
	string File = "./backlit/HISTOGRAM_B.jpg";
	float MAX;
	float hist_copy[256];
	Name = Name + ".jpg";
	Mat hist_dis = imread(File, IMREAD_UNCHANGED);
	if (hist_dis.empty())
	{
		cout << "HISTOGRAM_B image is missing in your folder" << endl;
		cout << "\n to view histogram images copy and paste HISTOGRAM_B image from input images to your cpp location folder" << endl;
		cin.get();
		exit(1);
	}

	Mat hist_copy_img = hist_dis.clone();
	MAX = hist[0];
	hist_copy[0] = hist[0];
	for (int i = 1; i < MAX_level; i++)
	{
		hist_copy[i] = hist[i];
		if (MAX < hist[i])
		{
			MAX = hist[i];
		}
	}

	// Adjusting the given histogram to a fixed size 512 X 512
	for (int i = 0; i < MAX_level; i++)
	{
		hist_copy[i] = (hist_copy[i] * 512) / MAX;
	}

	// HISTOGRAM display is done here
	for (int i = 511; i >= 0; i--)
	{
		for (int j = 0; j < 512; j = j + 2)
		{
			if ((int)hist_copy[j / 2] > 0)
			{
				hist_copy[j / 2] = hist_copy[j / 2] - 1;
				hist_copy_img.data[(i * 512) + j] = BRIGHTNESS;
				hist_copy_img.data[(i * 512) + j + 1] = BRIGHTNESS;
			}
		}
	}
	imshow(Name, hist_copy_img);
	Name = Name + ".jpg";
	//imwrite(Name, hist_copy_img);
}


//
//int LDA_thershold(float* hist)
//{
//	int* hist_copy = new int[MAX_level];
//	for (int i = 0; i < MAX_level; i++)
//	{
//		hist_copy[i] = (int)hist[i];
//	}
//	int bins_num = MAX_level;
//
//	// Calculate the bin_edges
//	long double bin_edges[256];
//	bin_edges[0] = 0.0;
//	long double increment = 0.99609375;
//	for (int i = 1; i < 256; i++)
//		bin_edges[i] = bin_edges[i - 1] + increment;
//
//	// Calculate bin_mids
//	long double bin_mids[256];
//	for (int i = 0; i < 256; i++)
//		bin_mids[i] = (bin_edges[i] + bin_edges[i + 1]) / 2;
//
//	// Iterate over all thresholds (indices) and get the probabilities weight1, weight2
//	long double weight1[256];
//	weight1[0] = hist_copy[0];
//	for (int i = 1; i < 256; i++)
//		weight1[i] = hist_copy[i] + weight1[i - 1];
//
//	int total_sum = 0;
//	for (int i = 0; i < 256; i++)
//		total_sum = total_sum + hist_copy[i];
//	long double weight2[256];
//	weight2[0] = total_sum;
//	for (int i = 1; i < 256; i++)
//		weight2[i] = weight2[i - 1] - hist_copy[i - 1];
//
//	// Calculate the class means: mean1 and mean2
//	long double histogram_bin_mids[256];
//	for (int i = 0; i < 256; i++)
//		histogram_bin_mids[i] = hist_copy[i] * bin_mids[i];
//
//	long double cumsum_mean1[256];
//	cumsum_mean1[0] = histogram_bin_mids[0];
//	for (int i = 1; i < 256; i++)
//		cumsum_mean1[i] = cumsum_mean1[i - 1] + histogram_bin_mids[i];
//
//	long double cumsum_mean2[256];
//	cumsum_mean2[0] = histogram_bin_mids[255];
//	for (int i = 1, j = 254; i < 256 && j >= 0; i++, j--)
//		cumsum_mean2[i] = cumsum_mean2[i - 1] + histogram_bin_mids[j];
//
//	long double mean1[256];
//	for (int i = 0; i < 256; i++)
//		mean1[i] = cumsum_mean1[i] / weight1[i];
//
//	long double mean2[256];
//	for (int i = 0, j = 255; i < 256 && j >= 0; i++, j--)
//		mean2[j] = cumsum_mean2[i] / weight2[j];
//
//	// Calculate Inter_class_variance
//	long double Inter_class_variance[255];
//	long double dnum = 10000000000;
//	for (int i = 0; i < 255; i++)
//		Inter_class_variance[i] = ((weight1[i] * weight2[i] * (mean1[i] - mean2[i + 1])) / dnum) * (mean1[i] - mean2[i + 1]);
//
//	// Maximize interclass variance
//	long double maxi = 0;
//	int getmax = 0;
//	for (int i = 0; i < 255; i++)
//	{
//		if (maxi < Inter_class_variance[i])
//		{
//			maxi = Inter_class_variance[i];
//			getmax = i;
//		}
//	}
//	cout << "\n\n\nOtsu's algorithm implementation thresholding result: " << bin_mids[getmax];
//
//	return ((int)bin_mids[getmax]);
//}


/*
* INPUT  -> Otsu threshold
*           number of rows and number of columns of input image
* OUTPUT -> Triangular histogram peak at Otsu threshold
* by equating the area of triangle and ROWS X COLUMNS we get height
* 1/2 X (TH X height) + 1/2 X ( (255-TH) X H) = ROWS X COLUMNS
* from (0,0) to (TH,height) slope is height/th
* from (TH,height) to (255,0) slope is height/(TH-255)
*/
void tri_hist(float* tar_hist, int th, int rows, int cols)
{
	float height = (rows * cols * 2) / 255;
	tar_hist[0] = 0;
	float max = (rows * cols);
	float a = height / th;
	for (int i = 1; i <= th; i++)
	{
		tar_hist[i] = (a * i);
	}
	a = 255 - th;
	int b = th * height;
	for (int i = th + 1; i < MAX_level; i++)
	{
		tar_hist[i] = height + (b / a) - ((height * i) / a);
	}
	HistogramDisplay(tar_hist, "TARGET_HIST_1");
}



/*
* INPUT  -> Otsu threshold
*           Triangular histogram
* OUTPUT -> Triangular histogram with area of B and F regions are equal with the input image area and peak at Otsu threshold
* Height of two right angled triangles will be different
* height 1 = (sum all pixels at B region / Threhold) X 2
* height 2 = (sum of all pixels at F region / (255 - threhold)) X 2
*/
void Target_hist2(float* tar2_hist, float* ref_hist, int th)
{
	float height1 = 0, height2 = 0;
	for (int i = 0; i <= th; i++)
	{
		height1 = height1 + ref_hist[i];
	}

	height1 = (height1 / th) * 2;

	for (int i = th + 1; i < 256; i++)
	{
		height2 = height2 + ref_hist[i];
	}
	height2 = (height2 / (255 - th)) * 2;

	tar2_hist[0] = 0;
	int a = height1 / th;
	for (int i = 1; i <= th; i++)
	{
		tar2_hist[i] = a * i;
	}
	a = 255 - th - 1;
	int b = (th + 1) * height2;
	for (int i = th + 1; i < MAX_level; i++)
	{
		tar2_hist[i] = height2 + (b / a) - ((height2 * i) / a);
	}
	HistogramDisplay(tar2_hist, "TARGET_HIST_2");
}



/*
* INPUT  -> Triangular histogram with height ratio equal with input image
*           Input image histogram
*           Otsu thrshold
* OUTPUT -> B - region histogram is taken from triangular histogram and F - region is taken from input image histograms
*/
void Target_hist3(float* tar3_hist, float* ref2, float* ref1, int th)
{
	for (int i = 0; i <= th; i++)
	{
		tar3_hist[i] = ref1[i];
	}
	for (int i = th + 1; i < MAX_level; i++)
	{
		tar3_hist[i] = ref2[i];
	}
	HistogramDisplay(tar3_hist, "TARGET_HIST_3");
}


/*
* INPUT  -> Image histogram
*           image matrix
* OUTPUT -> histogram => Normalised histogram => PDF histogram => CDF histogram => CDF histogram X 255
*/
void Function_CDF(float* histogram, Mat image, float* Shist)
{

	float pdf_hist[MAX_level], cdf_hist[MAX_level];
	long int rowXcol = image.rows * image.cols;
	pdf_hist[0] = histogram[0] / rowXcol;
	cdf_hist[0] = pdf_hist[0];
	Shist[0] = 255 * cdf_hist[0];

	for (int i = 1; i < 256; i++)
	{
		pdf_hist[i] = (float)histogram[i] / rowXcol;
		cdf_hist[i] = pdf_hist[i] + cdf_hist[i - 1];
		Shist[i] = 255 * cdf_hist[i];
	}

}



/*
* INPUT  -> Input histogram
*           Target histogram
*           number of ROWS and COLUMNS
* OUTPUT -> Matched histogram
*           Matched intensity values
*/
float** Histogram_Matching(float* input, float* target, int rows, int cols, float** IN_inten)
{
	int i, j;
	int* matched = new int[MAX_level];
	for (i = 0; i < MAX_level; i++)
	{
		for (j = 0; j < MAX_level && target[j] < input[i]; j++)
		{

		}
		if (j < 256)
		{
			matched[i] = j;
		}
		else
		{
			matched[i] = 255;
		}

	}
	// Memory assigning to temp array
	float** temp = new float* [rows];
	for (int i = 0; i < rows; i++)
	{
		temp[i] = new float[cols];
	}

	// temp is updated with matched intensity values
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			temp[i][j] = matched[(int)IN_inten[i][j]];
		}
	}
	return temp;
}


/*
* INPUT  -> Image histogram
*           Number of ROWS and COLUMNS
* OUTPUT -> Mean of image intensity values
*/
int MeanFUN(float* IN_hist, int rows, int cols)
{
	long long int sum = 0;
	for (int i = 1; i < MAX_level; i++)
	{
		sum = sum + ((int)IN_hist[i] * i);
	}
	sum = sum / (rows * cols);
	return (int)sum;
}

/*
* INPUT  -> Two variables
* OUTPUT -> return minimum value among those two
*/
float MINfun(float a, float b)
{
	if (a < b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

/*
* INPUT  -> Mean of input image intensity values
*           Gaussian blured input image
*           Histogram matched image intensities
*           Iput image intensities
* OUTPUT -> new intensities = ALPHA(Matched intensities) + (1 - ALPHA) X (Input intensities)
*           ALPHA = MIN(1,( (Gaussian image intensities - MEAN) / MEAN ))
*/
float** IdashFUN(int I, Mat gauss, float** matched, float** IN_inten)
{
	float alpa;
	float mod, mod1, mod2, one = 1.00000;
	int index = 0;
	int index_val;

	// Memory assignment for temp variable
	float** temp = new float* [gauss.rows];
	for (int i = 0; i < gauss.rows; i++)
	{
		temp[i] = new float[gauss.cols];
	}

	// IDASH intensities caluclations
	for (int i = 0; i < gauss.rows; i++)
	{
		for (int j = 0; j < gauss.cols; j++)
		{
			index = i * gauss.cols + j;
			index_val = (uint)gauss.data[index];
			if (index_val < I)
			{
				mod1 = I - index_val;
			}
			else
			{
				mod1 = index_val - I;
			}
			mod = mod1 / I;
			alpa = MINfun(one, mod);
			temp[i][j] = ((alpa * matched[i][j]) + ((1 - alpa) * IN_inten[i][j]));
			//cout << (int)mod1 << " "<< mod << endl;
			//cout << " " << alpa;
		}
	}

	return temp;
}



/*
* INPUT  -> Input image matrix
*           IDASH image intensites
*           Input image intensities
* OUTPUT -> HUE preserved image intensites by
*           new pexel value = (IDASH intensities / Input image intensities) Input image pixele value
*/
Mat Hue_pre(Mat IN, float** Idash, float** IN_inten)
{
	Mat out = IN.clone();
	int pre = 0;
	float val;
	float div = 0;
	/*if (out.channels() == NUM_channels)
	{*/
		for (int i = 0; i < out.rows; i++)
		{
			for (int j = 0; j < out.cols; j++)
			{
				pre = i * out.cols * NUM_channels + j * NUM_channels;
				for (int k = 0; k < 3; k++)
				{
					if (IN_inten[i][j] == 0)
					{
						out.data[pre + k] = (uint)IN.data[pre + k];
					}
					else
					{
						div = Idash[i][j] / IN_inten[i][j];
						val = (floor)(div * (uint)IN.data[pre + k]);
						if (val < 256)
						{
							out.data[pre + k] = val;
						}
						else
						{
							out.data[pre + k] = 255;
						}
					}
				}
			}
		}
	//}
	/*else
	{
		for (int i = 0; i < out.rows; i++)
		{
			for (int j = 0; j < out.cols; j++)
			{
				div = Idash[i][j] / IN_inten[i][j];
				out.data[i * out.cols + j] = (floor)(div * IN.data[i * out.cols + j]);
			}
		}
	}*/

	return out;
}


/*
* INPUT  -> Input image matrix
*           HUE preserved image is HSV converted
* OUTPUT -> Saturation preserved image
* Saturation feild of HUE preserved image is replaced with the input image HSV saturation
*/
Mat Saturation_matching(Mat IN, Mat sat)
{
	Mat Out;
	int rows = IN.rows;
	int cols = IN.cols;
	int colsXcha = cols * IN.channels();
	int index;
	int B, G, R, a, b, c;
	double sat_val;
	int dij;
	Mat IN_sat;
	cvtColor(IN, IN_sat, COLOR_BGR2HSV);
	if (IN.channels() == 3)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < colsXcha; j = j + 3)
			{
				index = i * colsXcha + j;
				B = IN.data[index];
				G = IN.data[index + 1];
				R = IN.data[index + 2];
				a = (B - R) * (B - R);
				b = (R - G) * (R - G);
				c = (G - B) * (G - B);
				sat_val = a + b + c;
				sat_val = sat_val / 3;
				dij = (uint)sqrt(sat_val);
				//cout << " "<<dij;
				//sat.data[index + 1] = dij;
				sat.data[index + 1] = (uint)IN_sat.data[index + 1];
			}
		}
	}
	cvtColor(sat, Out, COLOR_HSV2BGR);
	return Out;
}




int main()
{
	// FOLDER of ibpput images
	string File = "./backlit/";
	string Name;
	// INPUT image name
	cout << "Enter the input image name\n -> ";
	cin >> Name;
	// concatinate image name with .jpg
	File = File + Name; // +".jpg";

	Mat IN_image = imread(File, IMREAD_UNCHANGED);
	if (IN_image.empty())
	{
		cout << "\nCould not open or find the image" << endl;
		cin.get();
		return -1;
	}
	// INPUT imagr properties
	printD(IN_image);


	float* IN_histogram = new float[MAX_level];

	//HISTOGRAM calculation of input image
	float** IN_intensity = Histogram_cal(IN_image, IN_histogram);
	HistogramDisplay(IN_histogram, "INPUT");
	int th;

	// OTSU threshold to th variable
	Mat src = imread(File, IMREAD_GRAYSCALE);
	Mat dst;
	double thresh = 0;
	double maxValue = 255;
	th = (int)threshold(src, dst, thresh, maxValue, THRESH_OTSU);
	cout << "\n\nnew thershold from function  " << th;

	// construction of TARGET histograms 
	float* Target1 = new float[MAX_level];
	tri_hist(Target1, th, IN_image.rows, IN_image.cols);
	float* Target2 = new float[MAX_level];
	Target_hist2(Target2, IN_histogram, th);
	float* Target3 = new float[MAX_level];
	Target_hist3(Target3, IN_histogram, Target2, th);


	// caluclating CDF of all histograms 
	float* IN_Shist = new float[MAX_level];
	Function_CDF(IN_histogram, IN_image, IN_Shist);
	float* tar1_Shist = new float[MAX_level];
	Function_CDF(Target1, IN_image, tar1_Shist);
	float* tar2_Shist = new float[MAX_level];
	Function_CDF(Target2, IN_image, tar2_Shist);
	float* tar3_Shist = new float[MAX_level];
	Function_CDF(Target3, IN_image, tar3_Shist);

	// HISTOGRAM MATCHING for three targets
	float** Tar1_matched = Histogram_Matching(IN_Shist, tar1_Shist, IN_image.rows, IN_image.cols, IN_intensity);
	float** Tar2_matched = Histogram_Matching(IN_Shist, tar2_Shist, IN_image.rows, IN_image.cols, IN_intensity);
	float** Tar3_matched = Histogram_Matching(IN_Shist, tar3_Shist, IN_image.rows, IN_image.cols, IN_intensity);

	// calculation MEAN intensite for input image
	int Imean = MeanFUN(IN_histogram, IN_image.rows, IN_image.cols);
	Mat gauss;
	// Gaussian BLUR for input image
	GaussianBlur(src, gauss, Size(15, 15), 1);

	// calculating IDASH intensities
	float** Idash1 = IdashFUN(Imean, gauss, Tar1_matched, IN_intensity);
	float** Idash2 = IdashFUN(Imean, gauss, Tar2_matched, IN_intensity);
	float** Idash3 = IdashFUN(Imean, gauss, Tar3_matched, IN_intensity);

	// HUE preservation
	Mat hue1 = Hue_pre(IN_image, Idash1, IN_intensity);
	Mat hue2 = Hue_pre(IN_image, Idash2, IN_intensity);
	Mat hue3 = Hue_pre(IN_image, Idash3, IN_intensity);

	// HUE preserved image is converted to HSV
	Mat sat1, sat2, sat3;
	cvtColor(hue1, sat1, COLOR_BGR2HSV);
	cvtColor(hue2, sat2, COLOR_BGR2HSV);
	cvtColor(hue3, sat3, COLOR_BGR2HSV);

	// SATURATION matching 
	Mat tar1_out, tar2_out, tar3_out;
	tar1_out = Saturation_matching(IN_image, sat1);
	tar2_out = Saturation_matching(IN_image, sat2);
	tar3_out = Saturation_matching(IN_image, sat3);

	// DISPLAYING all outputs histograms and output images
	imshow("1", tar1_out);
	imshow("2", tar2_out);
	imshow("3", tar3_out);
	//imwrite("TARGET1.jpg", tar1_out);
	//imwrite("TARGET2.jpg", tar2_out);
	//imwrite("TARGET3.jpg", tar3_out);
	float* hist1 = new float[256];
	Histogram_cal(tar1_out, hist1);
	float* hist2 = new float[256];
	Histogram_cal(tar2_out, hist2);
	float* hist3 = new float[256];
	Histogram_cal(tar3_out, hist3);
	HistogramDisplay(hist1, "OUT_1");
	HistogramDisplay(hist2, "OUT_2");
	HistogramDisplay(hist3, "OUT_3");

	// INPUT image display
	imshow("INPUT_IMAGE", IN_image);

	waitKey(0);
	return 0;
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file