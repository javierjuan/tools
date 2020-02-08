/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2018 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* CImg <-> MATLAB data type conversions                                    *
***************************************************************************/

#ifndef EIGENITK_HPP
#define EIGENITK_HPP

#include <stdexcept>
#include <Eigen/Dense>
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#define __STR_FUNCNAME__  std::string(__FUNCTION__)

using namespace Eigen;

class EigenITK
{
public:
	template<typename T, int ImageDimensions, int MaskDimensions>
	static Matrix<T, Dynamic, Dynamic> toEigen(const typename itk::Image<T, ImageDimensions>::Pointer &image, const typename itk::Image<bool, MaskDimensions>::Pointer &mask);
	template <int ImageDimensions, int MaskDimensions, typename Derived>
	static typename itk::Image<typename Derived::Scalar, ImageDimensions>::Pointer toITK(const DenseBase<Derived> &data, const typename itk::Image<bool, MaskDimensions>::Pointer &mask);
	template <int ImageDimensions, int MaskDimensions, typename Derived>
	static void toITK(const DenseBase<Derived> &data, const typename itk::Image<bool, MaskDimensions>::Pointer &mask, typename itk::Image<typename Derived::Scalar, ImageDimensions>::Pointer &image);
};

template<typename T, int ImageDimensions, int MaskDimensions>
Matrix<T, Dynamic, Dynamic> EigenITK::toEigen(const typename itk::Image<T, ImageDimensions>::Pointer &image, const typename itk::Image<bool, MaskDimensions>::Pointer &mask)
{
	typedef itk::Image<T, ImageDimensions> ImageType;
	typedef itk::Image<bool, MaskDimensions> MaskType;

	typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
	typename MaskType::SizeType  maskSize  =  mask->GetLargestPossibleRegion().GetSize();
	
	if (imageSize[0] != maskSize[0] || imageSize[1] != maskSize[1])
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Image and mask dimensions must agree." << std::endl;
		throw std::runtime_error(s.str());
	}
	
	if ((ImageDimensions == MaskDimensions) || ((ImageDimensions - 1) == MaskDimensions))
	{
		for (int i = 0; i < MaskDimensions; ++i)
		{
			if (imageSize[i] != maskSize[i])
			{
				std::stringstream s;
				s << __STR_FUNCNAME__ << " ==> Image and mask dimensions must agree." << std::endl;
				throw std::runtime_error(s.str());
			}
		}
	}
	else
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Image dimensions should be equal or equal + 1 to mask dimensions." << std::endl;
		throw std::runtime_error(s.str());
	}
	
	int i = 0;
	int rows = 1;
	int cols = ImageDimensions == MaskDimensions ? 1 : imageSize[MaskDimensions];
	for (int j = 0; j < MaskDimensions; ++j)
	{
		rows *= imageSize[j];
	}
	
	Matrix<T, Dynamic, Dynamic> data(rows, cols);
	
	itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
	iterator.GoToBegin();
	
	while(!iterator.IsAtEnd())
	{
		if (iterator.Get())
		{
			itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
			itk::Index<ImageDimensions> imageIndex;
			
			for (int j = 0; j < MaskDimensions; ++j)
			{
				imageIndex[j] = maskIndex[j];
			}
			for (int j = 0; j < imageSize[MaskDimensions]; ++j)
			{
				imageIndex[MaskDimensions] = j;
				data(i, j) = image->GetPixel(imageIndex);
				
			}
			++i;
		}
		++iterator;
	}
	data.conservativeResize(i, data.cols());
	return data;
}

template <int ImageDimensions, int MaskDimensions, typename Derived>
void EigenITK::toITK(const DenseBase<Derived> &data, const typename itk::Image<bool, MaskDimensions>::Pointer &mask, typename itk::Image<typename Derived::Scalar, ImageDimensions>::Pointer &image)
{
	typedef itk::Image<typename Derived::Scalar, ImageDimensions> ImageType;
	typedef itk::Image<bool, MaskDimensions> MaskType;

	itk::ImageRegionConstIterator<MaskType> it(mask, mask->GetLargestPossibleRegion());
	it.GoToBegin();
	int count = 0;
	while (!it.IsAtEnd())
	{
		if (it.Get())
			++count;
		++it;
	}
	
	if (count != data.rows())
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Number of positive mask elements must agree with data rows." << std::endl;
		throw std::runtime_error(s.str());
	}

	typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();;

	itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
	iterator.GoToBegin();
	int i = 0;
	while (!iterator.IsAtEnd())
	{
		itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
		itk::Index<ImageDimensions> imageIndex;
		
		for (int j = 0; j < MaskDimensions; ++j)
		{
			imageIndex[j] = maskIndex[j];
		}
		
		if (iterator.Get())
		{
			for (int j = 0; j < imageSize[MaskDimensions]; j++)
			{
				imageIndex[MaskDimensions] = j;
				image->SetPixel(imageIndex, data(i, j));
			}
			++i;
		}
		else
		{
			for (int j = 0; j < imageSize[MaskDimensions]; j++)
			{
				imageIndex[MaskDimensions] = j;
				image->SetPixel(imageIndex, (typename Derived::Scalar) 0);

			}
		}
		++iterator;
	}
}


template <int ImageDimensions, int MaskDimensions, typename Derived>
typename itk::Image<typename Derived::Scalar, ImageDimensions>::Pointer EigenITK::toITK(const DenseBase<Derived> &data, const typename itk::Image<bool, MaskDimensions>::Pointer &mask)
{
	typedef itk::Image<typename Derived::Scalar, ImageDimensions> ImageType;
	typedef itk::Image<bool, MaskDimensions> MaskType;
	typedef itk::ImageRegion<ImageDimensions> RegionType;

	typename MaskType::SizeType maskSize = mask->GetLargestPossibleRegion().GetSize();
	typename RegionType::SizeType imageSize;
	
	for (int i = 0; i < MaskDimensions; ++i)
	{
		imageSize[i] = maskSize[i];
	}
	imageSize[MaskDimensions] = data.cols();
	RegionType imageRegion(imageSize);

	typename ImageType::Pointer image = ImageType::New();
	image->SetRegions(imageRegion);
	image->Allocate();

	EigenITK::toITK<ImageDimensions, MaskDimensions, Derived>(data, mask, image);

	return image;
}

#endif