/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2018 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* ITK utils                                                                *
***************************************************************************/

#ifndef ITKUTILS_HPP
#define ITKUTILS_HPP

#include <stdexcept>
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkChangeInformationImageFilter.h>

#define __STR_FUNCNAME__  std::string(__FUNCTION__)

class ITKUtils
{
public:
    template<typename T, int ImageDimensions, int MaskDimensions>
    static typename itk::Image<bool, MaskDimensions>::Pointer ZerosMask(const typename itk::Image<T, ImageDimensions>::Pointer &image);
    template<typename T, int ImageDimensions, int MaskDimensions>
    static typename itk::Image<bool, MaskDimensions>::Pointer NonZerosMask(const typename itk::Image<T, ImageDimensions>::Pointer &image);
    template<typename T, int ImageDimensions, int MaskDimensions>
    static typename itk::Image<bool, MaskDimensions>::Pointer NonZerosMaskIntersect(const typename itk::Image<T, ImageDimensions>::Pointer &image, const typename itk::Image<bool, MaskDimensions>::Pointer &mask);
};


template<typename SourceType, typename DestinationType>
void changeInformationImage(const typename SourceType::Pointer &source, typename DestinationType::Pointer &destination)
{
    using ChangeInfoFilterType = itk::ChangeInformationImageFilter<DestinationType>;

    typename ChangeInfoFilterType::Pointer changeFilter = ChangeInfoFilterType::New();
    changeFilter->SetOutputSpacing(source->GetSpacing());
    changeFilter->ChangeSpacingOn();
    changeFilter->SetOutputOrigin(source->GetOrigin());
    changeFilter->ChangeOriginOn();
    changeFilter->SetOutputDirection(source->GetDirection());
    changeFilter->ChangeDirectionOn();
    changeFilter->SetInput(destination);
    changeFilter->Update();
}


template<typename T, int ImageDimensions, int MaskDimensions>
typename itk::Image<bool, MaskDimensions>::Pointer ITKUtils::ZerosMask(const typename itk::Image<T, ImageDimensions>::Pointer &image)
{
    if ((MaskDimensions != ImageDimensions) && (MaskDimensions != (ImageDimensions - 1)))
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Image dimensions not consistent with mask dimensions. They should be equal or mask dimensions = image dimensions - 1." << std::endl;
        throw std::runtime_error(s.str());
    }

    using ImageType = itk::Image<T, ImageDimensions>;
    using MaskType = itk::Image<bool, MaskDimensions>;
    using RegionType = itk::ImageRegion<MaskDimensions> ;
    using ChangeInfoFilterType = itk::ChangeInformationImageFilter<MaskType>;

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    typename RegionType::SizeType regionSize;

    for (int i = 0; i < MaskDimensions; ++i)
    {
        regionSize[i] = imageSize[i];
    }
    RegionType maskRegion(regionSize);

    typename MaskType::Pointer mask = MaskType::New();
    mask->SetRegions(maskRegion);
    mask->Allocate();

    itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
    iterator.GoToBegin();
    if (ImageDimensions == MaskDimensions)
    {
        while (!iterator.IsAtEnd())
        {
            itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimensions> imageIndex;
            
            for (int i = 0; i < MaskDimensions; ++i)
            {
                imageIndex[i] = maskIndex[i];
            }
            mask->SetPixel(maskIndex, image->GetPixel(imageIndex) == (typename ImageType::PixelType) 0);
            ++iterator;
        }
    }
    else
    {
        while (!iterator.IsAtEnd())
        {
            bool flag = true;
            itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimensions> imageIndex;
            
            for (int i = 0; i < MaskDimensions; ++i)
            {
                imageIndex[i] = maskIndex[i];
            }
            for (int i = 0; i < imageSize[ImageDimensions]; ++i)
            {
                imageIndex[ImageDimensions] = i;
                flag = flag && (image->GetPixel(imageIndex) == (typename ImageType::PixelType) 0);
                if (!flag)
                    break;
            }
            mask->SetPixel(maskIndex, flag);
            ++iterator;
        }
    }
    return changeInformationImage(image, mask);
}

template<typename T, int ImageDimensions, int MaskDimensions>
typename itk::Image<bool, MaskDimensions>::Pointer ITKUtils::NonZerosMask(const typename itk::Image<T, ImageDimensions>::Pointer &image)
{
    if ((MaskDimensions != ImageDimensions) && (MaskDimensions != (ImageDimensions - 1)))
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Input dimensions not consistent with output dimensions. They should be equal or output dimensions = input dimensions - 1." << std::endl;
        throw std::runtime_error(s.str());
    }

    typedef itk::Image<T, ImageDimensions> ImageType;
    typedef itk::Image<bool, MaskDimensions> MaskType;
    typedef itk::ImageRegion<MaskDimensions> RegionType;

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    typename RegionType::SizeType regionSize;
    
    for (int i = 0; i < MaskDimensions; ++i)
    {
        regionSize[i] = imageSize[i];
    }
    
    RegionType maskRegion(regionSize);

    typename MaskType::Pointer mask = MaskType::New();
    mask->SetRegions(maskRegion);
    mask->Allocate();

    itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
    iterator.GoToBegin();
    if (ImageDimensions == MaskDimensions)
    {
        while (!iterator.IsAtEnd())
        {
            itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimensions> imageIndex;
            
            for (int i = 0; i < MaskDimensions; ++i)
            {
                imageIndex[i] = maskIndex[i];
            }
            mask->SetPixel(maskIndex, image->GetPixel(imageIndex) != (typename ImageType::PixelType) 0);
            ++iterator;
        }
    }
    else
    {
        while (!iterator.IsAtEnd())
        {
            bool flag = true;
            itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimensions> imageIndex;
            
            for (int i = 0; i < MaskDimensions; ++i)
            {
                imageIndex[i] = maskIndex[i];
            }
            for (int i = 0; i < imageSize[ImageDimensions]; ++i)
            {
                imageIndex[ImageDimensions] = i;                
                flag = flag && (image->GetPixel(imageIndex) != (typename ImageType::PixelType) 0);
                if (!flag)
                    break;
            }
            mask->SetPixel(maskIndex, flag);
            ++iterator;
        }
    }
    return changeInformationImage(image, mask);
}

template<typename T, int ImageDimensions, int MaskDimensions>
typename itk::Image<bool, MaskDimensions>::Pointer ITKUtils::NonZerosMaskIntersect(const typename itk::Image<T, ImageDimensions>::Pointer &image, const typename itk::Image<bool, MaskDimensions>::Pointer &mask)
{
    if ((MaskDimensions != ImageDimensions) && (MaskDimensions != (ImageDimensions - 1)))
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Input dimensions not consistent with output dimensions. They should be equal or output dimensions = input dimensions - 1." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    typedef itk::Image<T, ImageDimensions> ImageType;
    typedef itk::Image<bool, MaskDimensions> MaskType;
    typedef itk::ImageRegion<MaskDimensions> RegionType;

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    typename RegionType::SizeType regionSize;
    
    for (int i = 0; i < MaskDimensions; ++i)
    {
        regionSize[i] = imageSize[i];
    }
    
    RegionType maskRegion(regionSize);

    typename MaskType::Pointer maskIntersect = MaskType::New();
    maskIntersect->SetRegions(maskRegion);
    maskIntersect->Allocate();

    itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(maskIntersect, maskIntersect->GetLargestPossibleRegion());
    iterator.GoToBegin();
    if (ImageDimensions == MaskDimensions)
    {
        while (!iterator.IsAtEnd())
        {
            itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimensions> imageIndex;
            
            if (!mask->GetPixel(maskIndex))
            {
                maskIntersect->SetPixel(maskIndex, false);
            }
            else
            {
                for (int j = 0; j < MaskDimensions; ++j)
                {
                    imageIndex[j] = maskIndex[j];
                }
                maskIntersect->SetPixel(maskIndex, image->GetPixel(imageIndex) != (typename ImageType::PixelType) 0);
            }
            ++iterator;
        }
    }
    else
    {
        while (!iterator.IsAtEnd())
        {
            bool flag = true;
            itk::Index<MaskDimensions> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimensions> imageIndex;
            
            if (!mask->GetPixel(maskIndex))
            {
                maskIntersect->SetPixel(maskIndex, false);
            }
            else
            {
                for (int j = 0; j < MaskDimensions; ++j)
                {
                    imageIndex[j] = maskIndex[j];
                }
                for (int j = 0; j < imageSize[ImageDimensions]; ++j)
                {
                    imageIndex[ImageDimensions] = j;                
                    flag = flag && (image->GetPixel(imageIndex) != (typename ImageType::PixelType) 0);
                    if (!flag)
                        break;
                }
                maskIntersect->SetPixel(maskIndex, flag);
            }
            ++iterator;
        }
    }
    changeInformationImage<MaskType, MaskType>(mask, maskIntersect);

    return maskIntersect;
}

#endif