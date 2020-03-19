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

#include <string>
#include <stdexcept>
#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkNiftiImageIO.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMaskImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkChangeInformationImageFilter.h>


class ITKUtils
{
public:
    static typename itk::ImageIOBase::Pointer ReadImageInformation(const std::string &filePath);
    template<typename ImageType>
    static typename ImageType::Pointer ReadNIfTIImage(const std::string &filePath);
    template<typename ImageType>
    static void WriteNIfTIImage(const typename ImageType::Pointer image, const std::string &filePath);
    template<typename ImageType, typename MaskType>
    static void AssertCompatibleImageAndMaskTypes();
    template<typename ImageType, typename MaskType>
    static void AssertCompatibleImageAndMaskSizes(const typename ImageType::Pointer image, const typename MaskType::Pointer mask);
    template<typename InputImageType, typename ReferenceImageType>
    static void AllocateRegionFromInput(typename InputImageType::Pointer input, const typename ReferenceImageType::Pointer reference, const unsigned int channels = 0);
    template<unsigned int InputDimension, unsigned int OutputDimension>
    static itk::Index<OutputDimension> GetSpatialIndex(const itk::Index<InputDimension> inputIndex);
    template<typename ImageType>
    static typename ImageType::Pointer ChangeInformationImage(const typename ImageType::Pointer image, const typename ImageType::Pointer reference);
    template<typename ImageType, typename ReferenceType>
    static typename ImageType::Pointer ChangeInformationImage(const typename ImageType::Pointer image, const typename ReferenceType::SpacingType spacing, const typename ReferenceType::PointType origin, const typename ReferenceType::DirectionType direction);
    template<typename ImageType, typename MaskType>
    static typename MaskType::Pointer ZerosMask(const typename ImageType::Pointer image, const bool inverse = false, const bool any = false, const double tolerance = -1);
    template<typename ImageType, typename MaskType>
    static typename MaskType::Pointer ZerosMaskIntersect(const typename ImageType::Pointer image, const typename MaskType::Pointer mask, const bool inverse = false, const bool any = false, const double tolerance = -1);
};


typename itk::ImageIOBase::Pointer ITKUtils::ReadImageInformation(const std::string &filePath)
{
    typename itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(filePath.c_str(), itk::ImageIOFactory::FileModeType::ReadMode);

    imageIO->SetFileName(filePath);
    imageIO->ReadImageInformation();
    return imageIO;
}


template<typename ImageType>
typename ImageType::Pointer ITKUtils::ReadNIfTIImage(const std::string &filePath)
{
    // NIfTI IO
    itk::NiftiImageIO::Pointer NIfTIIO = itk::NiftiImageIO::New();

    // Typedefs
    using ImageReaderType = itk::ImageFileReader<ImageType>;

    // Read image
    typename ImageType::Pointer image;
    typename ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetImageIO(NIfTIIO);
    reader->SetFileName(filePath);
    reader->Update();
    return reader->GetOutput();
}


template<typename ImageType>
void ITKUtils::WriteNIfTIImage(const typename ImageType::Pointer image, const std::string &filePath)
{
    // NIfTI IO
    itk::NiftiImageIO::Pointer NIfTIIO = itk::NiftiImageIO::New();

    // Typedefs
    using ImageWriterType = itk::ImageFileWriter<ImageType>;

    // Write image
    typename ImageWriterType::Pointer writer = ImageWriterType::New();
    writer->SetImageIO(NIfTIIO);
    writer->SetInput(image);
    writer->SetFileName(filePath);
    writer->Update();
}


template<typename ImageType, typename MaskType>
void ITKUtils::AssertCompatibleImageAndMaskTypes()
{
    const unsigned int ImageDimension = ImageType::ImageDimension;
    const unsigned int MaskDimension = MaskType::ImageDimension;

    if ((ImageDimension < 2 || ImageDimension > 4) && (MaskDimension < 2 || MaskDimension > 3))
    {
        std::stringstream s;
        s << "Unsupported image dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }

    if ((MaskDimension != ImageDimension) && (MaskDimension != (ImageDimension - 1)))
    {
        std::stringstream s;
        s << "Incompatible image and mask dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<typename ImageType, typename MaskType>
void ITKUtils::AssertCompatibleImageAndMaskSizes(const typename ImageType::Pointer image, const typename MaskType::Pointer mask)
{
    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    typename MaskType::SizeType maskSize =  mask->GetLargestPossibleRegion().GetSize();

    for (int i = 0; i < MaskType::ImageDimension; ++i)
    {
        if (imageSize[i] != maskSize[i])
        {
            std::stringstream s;
            s << "Incompatible image and mask sizes" << std::endl;
            throw std::runtime_error(s.str());
        }
    }
}


template<typename InputImageType, typename ReferenceImageType>
void ITKUtils::AllocateRegionFromInput(typename InputImageType::Pointer input, const typename ReferenceImageType::Pointer reference, const unsigned int channels)
{
    const int InputDimension = InputImageType::ImageDimension;
    const int ReferenceDimension = ReferenceImageType::ImageDimension;

    if ((InputDimension != ReferenceDimension) && (std::abs(InputDimension - ReferenceDimension) != 1))
    {
        std::stringstream s;
        s << "Incompatible input and reference image dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }

    using InputRegionType = itk::ImageRegion<InputDimension>;

    typename ReferenceImageType::SizeType referenceSize = reference->GetLargestPossibleRegion().GetSize();
    typename InputRegionType::SizeType inputSize;
    inputSize.Fill(0);

    const int ChannelsDimensionIndex = InputDimension < ReferenceDimension ? InputDimension : ReferenceDimension;
    for (int i = 0; i < ChannelsDimensionIndex; ++i)
        inputSize[i] = referenceSize[i];

    if (channels > 0)
        inputSize[ChannelsDimensionIndex] = channels;

    InputRegionType inputRegion(inputSize);
    input->SetRegions(inputRegion);
    input->Allocate();
}


template<unsigned int InputDimension, unsigned int OutputDimension>
itk::Index<OutputDimension> ITKUtils::GetSpatialIndex(const itk::Index<InputDimension> inputIndex)
{
    itk::Index<OutputDimension> outputIndex;
    outputIndex.Fill(0);
    for (int i = 0; i < InputDimension; ++i)
        outputIndex[i] = inputIndex[i];
    return outputIndex;
}


template<typename ImageType>
typename ImageType::Pointer ITKUtils::ChangeInformationImage(const typename ImageType::Pointer image, const typename ImageType::Pointer reference)
{
    using ChangeInformationImageFilterType = itk::ChangeInformationImageFilter<ImageType>;

    typename ChangeInformationImageFilterType::Pointer changeInformationImageFilter = ChangeInformationImageFilterType::New();

    changeInformationImageFilter->SetReferenceImage(reference);
    changeInformationImageFilter->UseReferenceImageOn();
    changeInformationImageFilter->SetInput(image);
    changeInformationImageFilter->Update();
    return changeInformationImageFilter->GetOutput();
}

template<typename ImageType, typename ReferenceType>
typename ImageType::Pointer ITKUtils::ChangeInformationImage(const typename ImageType::Pointer image, const typename ReferenceType::SpacingType spacing, const typename ReferenceType::PointType origin, const typename ReferenceType::DirectionType direction)
{
    const unsigned int ImageDimension = ImageType::ImageDimension;
    const unsigned int ReferenceDimension = ReferenceType::ImageDimension;

    if (ImageDimension > ReferenceDimension)
    {
        std::stringstream s;
        s << "Incompatible input and reference image dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }

    using ChangeInformationImageFilterType = itk::ChangeInformationImageFilter<ImageType>;

    typename ChangeInformationImageFilterType::Pointer changeInformationImageFilter = ChangeInformationImageFilterType::New();

    typename ImageType::SpacingType imageSpacing;
    typename ImageType::PointType imageOrigin;
    typename ImageType::DirectionType imageDirection;

    for (int i = 0; i < ImageDimension; ++i)
    {
        imageSpacing[i] = spacing[i];
        imageOrigin[i] = origin[i];
        for (int j = 0; j < ImageDimension; ++j)
        {
            imageDirection[i][j] = direction[i][j];
        }
    }
    changeInformationImageFilter->SetOutputSpacing(imageSpacing);
    changeInformationImageFilter->ChangeSpacingOn();
    changeInformationImageFilter->SetOutputOrigin(imageOrigin);
    changeInformationImageFilter->ChangeOriginOn();
    changeInformationImageFilter->SetOutputDirection(imageDirection);
    changeInformationImageFilter->ChangeDirectionOn();
    changeInformationImageFilter->SetInput(image);
    changeInformationImageFilter->Update();
    return changeInformationImageFilter->GetOutput();
}


template<typename ImageType, typename MaskType>
typename MaskType::Pointer ITKUtils::ZerosMask(const typename ImageType::Pointer image, const bool inverse, const bool any, const double tolerance)
{
    ITKUtils::AssertCompatibleImageAndMaskTypes<ImageType, MaskType>();

    const unsigned int ImageDimension = ImageType::ImageDimension;
    const unsigned int MaskDimension = MaskType::ImageDimension;

    typename MaskType::Pointer mask = MaskType::New();
    ITKUtils::AllocateRegionFromInput<MaskType, ImageType>(mask, image);

    typename ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    itk::ImageRegionIteratorWithIndex<MaskType> iterator(mask, mask->GetLargestPossibleRegion());
    iterator.GoToBegin();
    while (!iterator.IsAtEnd())
    {
        itk::Index<MaskDimension> maskIndex = iterator.GetIndex();
        itk::Index<ImageDimension> imageIndex = ITKUtils::GetSpatialIndex<MaskDimension, ImageDimension>(maskIndex);

        const typename ImageType::PixelType value = image->GetPixel(imageIndex);

        if (ImageDimension == MaskDimension)
            iterator.Set(inverse ? value != 0 : value == 0);
        else
        {
            bool flag = any;
            unsigned int count = 0;
            for (int i = 0; i < imageSize[MaskDimension]; ++i)
            {
                imageIndex[ImageDimension - 1] = i;
                const typename ImageType::PixelType value = image->GetPixel(imageIndex);
                const bool condition = (inverse ? value != 0 : value == 0);
                count += (unsigned int) condition;
                if (tolerance > 0 && tolerance < 1)
                    flag = ((double) count / (double) imageSize[MaskDimension]) >= (1.0 - tolerance);
                else
                {
                    flag = any ? flag && condition : flag || condition;
                    if (!flag)
                        break;
                }
            }
            iterator.Set(flag);
        }
        ++iterator;
    }
    return ITKUtils::ChangeInformationImage<MaskType, ImageType>(mask, image->GetSpacing(), image->GetOrigin(), image->GetDirection());
}


template<typename ImageType, typename MaskType>
typename MaskType::Pointer ITKUtils::ZerosMaskIntersect(const typename ImageType::Pointer image, const typename MaskType::Pointer mask, const bool inverse, const bool any, const double tolerance)
{
    using MaskFilterType = itk::MaskImageFilter<MaskType, MaskType>;

    typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();
    maskFilter->SetInput(ITKUtils::ZerosMask<ImageType, MaskType>(image, inverse, any, tolerance));
    maskFilter->SetMaskImage(mask);
    maskFilter->Update();
    return maskFilter->GetOutput();
}

#endif