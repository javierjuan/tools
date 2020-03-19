/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* CImg utils                                                               *
***************************************************************************/

#ifndef CIMGUTILS_HPP
#define CIMGUTILS_HPP

#define cimg_display 0
#include <CImg.h>
#include <stdexcept>
#include <sstream>
#include <omp.h>

using namespace cimg_library;

class CImgUtils
{
public:
    template <typename T>
    static CImg<bool> ZerosMask(const CImg<T> &image);
    template <typename T>
    static CImg<bool> NonZerosMask(const CImg<T> &image);
    template <typename T>
    static CImg<bool> NonZerosMaskIntersect(const CImg<T> &image, const CImg<bool> &mask);
};

template <typename T>
CImg<bool> CImgUtils::ZerosMask(const CImg<T> &image)
{
    CImg<bool> mask(image.width(), image.height(), image.depth());
    #pragma omp parallel for
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                bool flag = true;
                for (int c = 0; c < image.spectrum(); ++c)
                    flag = flag && (image(x, y, z, c) == 0);
                mask(x, y, z) = flag;
            }
        }
    }
    return mask;
}

template <typename T>
CImg<bool> CImgUtils::NonZerosMask(const CImg<T> &image)
{
    CImg<bool> mask(image.width(), image.height(), image.depth());
    #pragma omp parallel for
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                bool flag = true;
                for (int c = 0; c < image.spectrum(); ++c)
                    flag = flag && (image(x, y, z, c) != 0);
                mask(x, y, z) = flag;
            }
        }
    }
    return mask;
}

template <typename T>
CImg<bool> CImgUtils::NonZerosMaskIntersect(const CImg<T> &image, const CImg<bool> &mask)
{
    if (!image.is_sameXYZ(mask))
    {
        std::stringstream s;
        s << "In function: " << __func__ << " ==> Image and mask dimensions must agree." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    CImg<bool> maskIntersect(image.width(), image.height(), image.depth());
    #pragma omp parallel for
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                if (!mask(x, y, z))
                    maskIntersect(x, y, z) = false;
                else
                {
                    bool flag = true;
                    for (int c = 0; c < image.spectrum(); ++c)
                        flag = flag && image(x, y, z, c) != 0;
                    maskIntersect(x, y, z) = flag;
                }
            }
        }
    }
    return maskIntersect;
}

#endif