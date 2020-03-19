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

#define cimg_display 0
#include <CImg.h>
#include <ITKUtils.hpp>
#include <stdexcept>
#include <itkImage.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIteratorWithIndex.h>

using namespace cimg_library;

class CImgITK
{
public:
    template<class ImageType>
    static CImg<typename ImageType::PixelType> toCImg(const typename ImageType::Pointer image);
    template<class ImageType>
    static typename ImageType::Pointer toITK(const CImg<typename ImageType::PixelType> &image);
};

template<class ImageType>
CImg<typename ImageType::PixelType> CImgITK::toCImg(const typename ImageType::Pointer image)
{
    const unsigned int ImageDimension = ImageType::ImageDimension;
    
    if (ImageDimension < 2 || ImageDimension > 4)
    {
        std::stringstream s;
        s << "Unsupported image dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    const unsigned int width = imageSize[0];
    const unsigned int height = imageSize[1];
    const unsigned int depth = ImageDimension > 2 ? imageSize[2] : 1;
    const unsigned int spectrum = ImageDimension > 3 ? imageSize[3] : 1;

    CImg<typename ImageType::PixelType> newImage(width, height, depth, spectrum);

    itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(image, image->GetLargestPossibleRegion());
    iterator.GoToBegin();
    while (!iterator.IsAtEnd())
    {
        itk::Index<ImageDimension> index = iterator.GetIndex();
        const unsigned int x = index[0];
        const unsigned int y = index[1];
        const unsigned int z = ImageDimension > 2 ? index[2] : 0;
        const unsigned int s = ImageDimension > 3 ? index[3] : 0;
        newImage(x, y, z, s) = iterator.Get();
        ++iterator;
    }
    return newImage;
}

template<class ImageType>
typename ImageType::Pointer CImgITK::toITK(const CImg<typename ImageType::PixelType> &image)
{
    const unsigned int ImageDimension = ImageType::ImageDimension;
    
    if (ImageDimension < 2 || ImageDimension > 4)
    {
        std::stringstream s;
        s << "Unsupported image dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if ((ImageDimension == 3 && image.depth() == 1) || (ImageDimension == 4 && (image.depth() == 1 || image.spectrum() == 1)))
    {
        std::stringstream s;
        s << "Incompatible image dimensions with templated ImageType" << std::endl;
        throw std::runtime_error(s.str());
    }

    using RegionType = itk::ImageRegion<ImageDimension>;
    typename RegionType::SizeType imageSize;
    
    imageSize[0] = image.width();
    imageSize[1] = image.height();
    if (ImageDimension > 2)
        imageSize[2] = image.depth();
    if (ImageDimension > 3)
        imageSize[3] = image.spectrum();
    
    RegionType imageRegion(imageSize);
    
    typename ImageType::Pointer newImage = ImageType::New();
    newImage->SetRegions(imageRegion);
    newImage->Allocate();
    
    itk::ImageRegionIteratorWithIndex<ImageType> iterator(newImage, newImage->GetLargestPossibleRegion());
    iterator.GoToBegin();
    while (!iterator.IsAtEnd())
    {
        itk::Index<ImageDimension> index = iterator.GetIndex();
        const unsigned int x = index[0];
        const unsigned int y = index[1];
        const unsigned int z = ImageDimension > 2 ? index[2] : 0;
        const unsigned int s = ImageDimension > 3 ? index[3] : 0;
        iterator.Set(image(x, y, z, s))
        ++iterator;
    }
    return newImage;
}

#endif