/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2018 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* CImg <-> ITK data type conversions                                       *
***************************************************************************/

#ifndef CIMGITK_HPP
#define CIMGITK_HPP

#include "mex.h"
#define cimg_display 0
#include <CImg.h>
#include <stdexcept>
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#define __STR_FUNCNAME__  std::string(__FUNCTION__)

using namespace cimg_library;

class CImgITK
{
public:
	template<typename T, int ImageDimensions>
	static CImg<T> toCImg(const typename itk::Image<T, ImageDimensions>::Pointer &image);
	template<typename T, int ImageDimensions>
	static typename itk::Image<T, ImageDimensions>::Pointer toITK(const CImg<T> &image);
};

template<typename T, int ImageDimensions>
static CImg<T> CImgITK::toCImg(const typename itk::Image<T, ImageDimensions>::Pointer &image)
{
	typedef itk::Image<T, ImageDimensions> ImageType;

	ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
	
	if (ImageDimensions < 2 || ImageDimensions > 4)
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Image supported dimensions are: 2, 3 or 4." << std::endl;
		throw std::runtime_error(s.str());
	}
	
	if (ImageDimensions == 2)
	{
		CImg<T> newImage(imageSize[0], imageSize[1]);
		itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(image, image->GetLargestPossibleRegion());
		iterator.GoToBegin();
		while (!iterator.IsAtEnd())
		{
			itk::Index<ImageDimensions> index = iterator.GetIndex();
			newImage(index[0], index[1]) = iterator.Get();
			++iterator;
		}
		return newImage;
	}
	else if (ImageDimensions == 3)
	{
		CImg<T> newImage(imageSize[0], imageSize[1], imageSize[2]);
		itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(image, image->GetLargestPossibleRegion());
		iterator.GoToBegin();
		while (!iterator.IsAtEnd())
		{
			itk::Index<ImageDimensions> index = iterator.GetIndex();
			newImage(index[0], index[1], index[2]) = iterator.Get();
			++iterator;
		}
		return newImage;
	}
	else if (ImageDimensions == 4)
	{
		CImg<T> newImage(imageSize[0], imageSize[1], imageSize[2], imageSize[3]);
		itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(image, image->GetLargestPossibleRegion());
		iterator.GoToBegin();
		while (!iterator.IsAtEnd())
		{
			itk::Index<ImageDimensions> index = iterator.GetIndex();
			newImage(index[0], index[1], index[2], imageSize[3]) = iterator.Get();
			++iterator;
		}
		return newImage;
	}
	else
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Image supported dimensions are: 2, 3 or 4." << std::endl;
		throw std::runtime_error(s.str());
	}
}

template<typename T, int ImageDimensions>
static typename itk::Image<T, ImageDimensions>::Pointer CImgITK::toITK(const CImg<T> &image)
{
	typedef itk::Image<T, ImageDimensions> ImageType;
	typedef itk::ImageRegion<ImageDimensions> RegionType;
	
	if (ImageDimensions < 2 || ImageDimensions > 4)
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Image supported dimensions are: 2, 3 or 4." << std::endl;
		throw std::runtime_error(s.str());
	}
	
	if ((ImageDimensions == 3 && image.depth() == 1) || (ImageDimensions == 4 && (image.depth() == 1 || image.spectrum() == 1)))
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Input image dimensions do not match with templated <ImageDimensions> parameter." << std::endl;
		throw std::runtime_error(s.str());
	}
	
	RegionType::SizeType imageSize;
	
	if (ImageDimensions == 2)
	{
		imageSize[0] = image.width();
		imageSize[1] = image.height();
		
		RegionType imageRegion(imageSize);
		
		ImageType::Pointer newImage = ImageType::New();
		newImage->SetRegions(imageRegion);
		newImage->Allocate();
		
		itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(newImage, newImage->GetLargestPossibleRegion());
		iterator.GoToBegin();
		while (!iterator.IsAtEnd())
		{
			itk::Index<ImageDimensions> index = iterator.GetIndex();
			newImage->SetPixel(index, image(index[0], index[1]));
			++iterator;
		}
		return newImage;
		
	}
	else if (ImageDimensions == 3)
	{
		imageSize[0] = image.width();
		imageSize[1] = image.height();
		imageSize[2] = image.depth();
		
		RegionType imageRegion(imageSize);
		
		ImageType::Pointer newImage = ImageType::New();
		newImage->SetRegions(imageRegion);
		newImage->Allocate();
		
		itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(newImage, newImage->GetLargestPossibleRegion());
		iterator.GoToBegin();
		while (!iterator.IsAtEnd())
		{
			itk::Index<ImageDimensions> index = iterator.GetIndex();
			newImage->SetPixel(index, image(index[0], index[1], index[2]));
			++iterator;
		}
		return newImage;
	}
	else if (ImageDimensions == 4)
	{
		imageSize[0] = image.width();
		imageSize[1] = image.height();
		imageSize[2] = image.depth();
		imageSize[3] = image.spectrum();
		
		RegionType imageRegion(imageSize);
		
		ImageType::Pointer newImage = ImageType::New();
		newImage->SetRegions(imageRegion);
		newImage->Allocate();
		
		itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(newImage, newImage->GetLargestPossibleRegion());
		iterator.GoToBegin();
		while (!iterator.IsAtEnd())
		{
			itk::Index<ImageDimensions> index = iterator.GetIndex();
			newImage->SetPixel(index, image(index[0], index[1], index[2], index[3]));
			++iterator;
		}
		return newImage;
	}
	else
	{
		std::stringstream s;
		s << __STR_FUNCNAME__ << " ==> Image supported dimensions are: 2, 3 or 4." << std::endl;
		throw std::runtime_error(s.str());
	}
}

#endif