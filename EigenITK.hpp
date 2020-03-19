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
#include <ITKUtils.hpp>
#include <itkImage.h>
#include <itkImageRegionConstIteratorWithIndex.h>

using namespace Eigen;

class EigenITK
{
public:
    template<typename ImageType, typename MaskType>
    static Matrix<typename ImageType::PixelType, Dynamic, Dynamic> toEigen(const typename ImageType::Pointer image, const typename MaskType::Pointer mask);
    template <typename ImageType, typename MaskType, typename Derived>
    static typename ImageType::Pointer toITK(const DenseBase<Derived> &data, const typename MaskType::Pointer mask);
    template <typename ImageType, typename MaskType, typename Derived>
    static void toITK(const DenseBase<Derived> &data, const typename MaskType::Pointer mask, typename ImageType::Pointer image);
};

template<typename ImageType, typename MaskType>
Matrix<typename ImageType::PixelType, Dynamic, Dynamic> EigenITK::toEigen(const typename ImageType::Pointer image, const typename MaskType::Pointer mask)
{
    ITKUtils::AssertCompatibleImageAndMaskTypes<ImageType, MaskType>();
    ITKUtils::AssertCompatibleImageAndMaskSizes<ImageType, MaskType>(image, mask);

    const unsigned int ImageDimension = ImageType::ImageDimension;
    const unsigned int MaskDimension = MaskType::ImageDimension;
    const unsigned int ChannelsDimensionIndex = ImageDimension - 1;

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    int rows = 1;
    int cols = ImageDimension == MaskDimension ? 1 : imageSize[ChannelsDimensionIndex];
    for (int j = 0; j < ChannelsDimensionIndex; ++j)
    {
        rows *= imageSize[j];
    }
    
    Matrix<typename ImageType::PixelType, Dynamic, Dynamic> data(rows, cols);
    
    itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
    int i = 0;
    iterator.GoToBegin();
    while(!iterator.IsAtEnd())
    {
        if (iterator.Get())
        {
            itk::Index<MaskDimension> maskIndex = iterator.GetIndex();
            itk::Index<ImageDimension> imageIndex = ITKUtils::GetSpatialIndex<MaskDimension, ImageDimension>(maskIndex);

            if (ImageDimension == MaskDimension)
                data(i, 0) = image->GetPixel(imageIndex);
            else
            {
                for (int j = 0; j < imageSize[ChannelsDimensionIndex]; ++j)
                {
                    imageIndex[ChannelsDimensionIndex] = j;
                    data(i, j) = image->GetPixel(imageIndex);
                    
                }
            }
            ++i;
        }
        ++iterator;
    }
    data.conservativeResize(i, data.cols());
    return data;
}

template <typename ImageType, typename MaskType, typename Derived>
typename ImageType::Pointer EigenITK::toITK(const DenseBase<Derived> &data, const typename MaskType::Pointer mask)
{
    ITKUtils::AssertCompatibleImageAndMaskTypes<ImageType, MaskType>();

    typename ImageType::Pointer image = ImageType::New();
    ITKUtils::AllocateRegionFromInput<ImageType, MaskType>(image, mask, data.cols());
    EigenITK::toITK<ImageType, MaskType, Derived>(data, mask, image);

    return image;
}

template <typename ImageType, typename MaskType, typename Derived>
void EigenITK::toITK(const DenseBase<Derived> &data, const typename MaskType::Pointer mask, typename ImageType::Pointer image)
{
    ITKUtils::AssertCompatibleImageAndMaskTypes<ImageType, MaskType>();
    ITKUtils::AssertCompatibleImageAndMaskSizes<ImageType, MaskType>(image, mask);

    const unsigned int ImageDimension = ImageType::ImageDimension;
    const unsigned int MaskDimension = MaskType::ImageDimension;
    const unsigned int ChannelsDimensionIndex = ImageDimension - 1;

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
        s << "Incompatible number of rows and positive elements in the mask" << std::endl;
        throw std::runtime_error(s.str());
    }

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();;

    itk::ImageRegionConstIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
    int i = 0;
    iterator.GoToBegin();
    while (!iterator.IsAtEnd())
    {
        itk::Index<MaskDimension> maskIndex = iterator.GetIndex();
        itk::Index<ImageDimension> imageIndex = ITKUtils::GetSpatialIndex<MaskDimension, ImageDimension>(maskIndex);

        if (ImageDimension == MaskDimension)
            image->SetPixel(imageIndex, (typename ImageType::PixelType) iterator.Get() ? data(i, 0) : 0);
        else
        {
            for (int j = 0; j < imageSize[ChannelsDimensionIndex]; j++)
            {
                imageIndex[ChannelsDimensionIndex] = j;
                image->SetPixel(imageIndex, (typename ImageType::PixelType) iterator.Get() ? data(i, j) : 0);
            }    
        }
        i += iterator.Get() ? 1 : 0;
        ++iterator;
    }
}

#endif