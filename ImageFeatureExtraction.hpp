/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2018 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Classic image feature extraction algorithms                              *
***************************************************************************/

#define cimg_display 0
#include <CImg.h>
#include <CImgMATLAB.hpp>
#include <EigenMATLAB.hpp>
#include <ScalarMATLAB.hpp>
#include <STDMATLAB.hpp>
#include <vector>
#include <cmath>
#include <omp.h>

#define INF 1e+300

using namespace cimg_library;


class ImageFeatureExtraction
{
public:
	template<typename T>
	static CImg<double> FirstOrderCentralMoments(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius);
	template<typename T>
	static CImg<double> LocalMedianMAD(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius);
	template<typename T>
	static CImg<double> LocalEnergy(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius);
	template<typename T>
	static CImg<double> LocalEntropy(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius, const int nbins);
}

template<typename T>
CImg<double> ImageFeatureExtraction::FirstOrderCentralMoments(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius)
{
	CImg<double> F(image.width(), image.height(), image.depth(), 7);
	
    #pragma omp parallel for
	for (int x = 0; x < image.width(); ++x)
	{
		for (int y = 0; y < image.height(); ++y)
		{
			for (int z = 0; z < image.depth(); ++z)
			{
				for (int f = 0; f < 7; ++f)
					F(x, y, z, f) = 0;
				
				if (!mask(x, y, z))	
					continue;
				
				// Local Mean
				int n = 0;
				double mu = 0;
				for (int xx = -radius[0]; xx <= radius[0]; ++xx)
				{
					const int xr = x + xx;
					if (xr < 0 || xr >= image.width())
						continue;
					
					for (int yy = -radius[1]; yy <= radius[1]; ++yy)
					{
						const int yr = y + yy;
						if (yr < 0 || yr >= image.height())
							continue;
						for (int zz = -radius[2]; zz <= radius[2]; ++zz)
						{
							const int zr = z + zz;
							if (zr < 0 || zr >= image.depth())
								continue;
							
							if (!mask(xr, yr, zr))
								continue;
							
							mu += image(xr, yr, zr);
							++n;
						}
					}
				}
				mu = n == 0 ? image(x, y, z) : mu / (double) n;
				
				// Variance, Skewness, Kurtosis, Energy0, Energy1
				n = 0;
				double var = 0;
				double skw = 0;
				double krt = 0;
				double energy0 = 0;
				double energy1 = 0;
				double max = -INF;
				double min = INF;
				for (int xx = -radius[0]; xx <= radius[0]; ++xx)
				{
					const int xr = x + xx;
					if (xr < 0 || xr >= image.width())
						continue;
					
					for (int yy = -radius[1]; yy <= radius[1]; ++yy)
					{
						const int yr = y + yy;
						if (yr < 0 || yr >= image.height())
							continue;
						
						for (int zz = -radius[2]; zz <= radius[2]; ++zz)
						{
							const int zr = z + zz;
							if (zr < 0 || zr >= image.depth())
								continue;
							
							if (!mask(xr, yr, zr))
								continue;
	
							const double value = image(xr, yr, zr);
							const double diff = value - mu;
							
							var += (diff * diff);
                            skw += (diff * diff * diff);
                            krt += (diff * diff * diff * diff);
							energy0 += (value * value);
							energy1 += value;
							max = value > max ? value : max;
							min = value < min ? value : min;
							++n;
						}
					}
				}
				F(x, y, z, 0) = mu;
				F(x, y, z, 1) = n <= 1 ? 0 : var / (double) (n - 1);
				F(x, y, z, 2) = n <= 0 || var == 0 ? 0 : (skw / (double) n) / pow(var / (double) n, 1.5);
				F(x, y, z, 3) = n <= 0 || var == 0 ? 0 : ((krt / (double) n) / pow(var / (double) n, 2)) - 3;
				F(x, y, z, 4) = n == 0 ? 0 : energy0 / (double) n;
				F(x, y, z, 5) = n == 0 ? 0 : (energy1 * energy1) / (double) (n * n);
				F(x, y, z, 6) = max - min;
			}
		}
	}
	return F;
}

template<typename T>
CImg<double> ImageFeatureExtraction::LocalEntropy(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius, const int nbins)
{
	CImg<double> F(image.width(), image.height(), image.depth());
	const double ratio = 256 / (double) nbins;
	
	#pragma omp parallel for
	for (int x = 0; x < image.width(); ++x)
	{
		for (int y = 0; y < image.height(); ++y)
		{
			for (int z = 0; z < image.depth(); ++z)
			{
				F(x, y, z) = 0;
				
				if (!mask(x, y, z))	
					continue;
				
				int n = 0;
				std::vector<int> bins(nbins, 0);
				for (int xx = -radius[0]; xx <= radius[0]; ++xx)
				{
					const int xr = x + xx;
					if (xr < 0 || xr >= image.width())
						continue;
					
					for (int yy = -radius[1]; yy <= radius[1]; ++yy)
					{
						const int yr = y + yy;
						if (yr < 0 || yr >= image.height())
							continue;
						
						for (int zz = -radius[2]; zz <= radius[2]; ++zz)
						{
							const int zr = z + zz;
							if (zr < 0 || zr >= image.depth())
								continue;
							
							if (!mask(xr, yr, zr))
								continue;
							
							bins[std::floor((double) image(xr, yr, zr) / ratio)]++;
							n++;
						}
					}
				}
				
				if (n == 0)
					continue;
				
				double H = 0;
				for (int i = 0; i < nbins; i++)
				{
					const double p = bins[i] / (double) n;
					if (p > 0)
						H += p * log(p);
				}
				F(x, y, z) = -H;
			}
		}
	}
	return F;
}

template<typename T>
CImg<double> ImageFeatureExtraction::LocalMedianMAD(const CImg<T> &image, const CImg<bool> &mask, const std::vector<int> &radius)
{
	const int N = (2 * radius[0] + 1) * (2 * radius[1] + 1) * (2 * radius[2] + 1);
	const int mid = (int) std::floor((double) N / 2.0);
	
	CImg<double> F(image.width(), image.height(), image.depth(), 2);
		
	#pragma omp parallel for
	for (int x = 0; x < image.width(); ++x)
	{
		for (int y = 0; y < image.height(); ++y)
		{
			for (int z = 0; z < image.depth(); ++z)
			{
				F(x, y, z, 0) = 0;
				F(x, y, z, 1) = 0;
				
				if (!mask(x, y, z))	
					continue;
				
				std::vector<double> patch;
				for (int xx = -radius[0]; xx <= radius[0]; ++xx)
				{
					const int xr = x + xx;
					if (xr < 0 || xr >= image.width())
					{
						patch.push_back(image(x, y, z));
						continue;
					}
					
					for (int yy = -radius[1]; yy <= radius[1]; ++yy)
					{
						const int yr = y + yy;
						if (yr < 0 || yr >= image.height())
						{
							patch.push_back(image(x, y, z));
							continue;
						}
						
						for (int zz = -radius[2]; zz <= radius[2]; ++zz)
						{
							const int zr = z + zz;
							if (zr < 0 || zr >= image.depth())
							{
								patch.push_back(image(x, y, z));
								continue;
							}
							
							patch.push_back(image(xr, yr, zr));
						}
					}
				}
				// Sort data and get median value
				std::sort(patch.begin(), patch.end());
				double median = patch[mid];
				F(x, y, z, 0) = median;
				
				// Compute MAD
				std::vector<double> residues;
				for (int l = 0; l < N; l++)
					residues.push_back(std::abs(patch[l] - median));
				
				// Sort data and get median value
				std::sort(residues.begin(), residues.end());
				F(x, y, z, 1) = residues[mid];
			}
		}
	}
	return F;
}
