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
    static CImg<bool> ZerosMask(const CImg<T> &image, const bool inverse = false, const double tolerance = 0);
    template <typename T>
    static CImg<bool> ZerosMaskIntersect(const CImg<T> &image, const CImg<bool> &mask, const bool inverse = false, const double tolerance = 0);
};

template <typename T>
CImg<bool> CImgUtils::ZerosMask(const CImg<T> &image, const bool inverse, const double tolerance)
{
    if (tolerance < 0 || tolerance > 1)
    {
        std::stringstream s;
        s << "In function: " << __func__ << " ==> Unexpected value for <tolerance>. It must be in the range [0,1]" << std::endl;
        throw std::runtime_error(s.str());
    }

    CImg<bool> mask(image.width(), image.height(), image.depth());
    #pragma omp parallel for
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                int count = 0;
                bool flag = false;
                for (int c = 0; c < image.spectrum(); ++c)
                {
                    const bool condition = (inverse ? image(x, y, z) != 0 : image(x, y, z) == 0);
                    count += (int) condition;
                    flag = ((double) count / (double) image.spectrum()) >= (1.0 - tolerance);
                    if (flag)
                        break;
                }
                mask(x, y, z) = flag;
            }
        }
    }
    return mask;
}

template <typename T>
CImg<bool> CImgUtils::ZerosMaskIntersect(const CImg<T> &image, const CImg<bool> &mask, const bool inverse, const double tolerance)
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
                    int count = 0;
                    bool flag = false;
                    for (int c = 0; c < image.spectrum(); ++c)
                    {
                        const bool condition = (inverse ? image(x, y, z) != 0 : image(x, y, z) == 0);
                        count += (int) condition;
                        flag = ((double) count / (double) image.spectrum()) >= (1.0 - tolerance);
                        if (flag)
                            break;
                    }
                    maskIntersect(x, y, z) = flag;
                }
            }
        }
    }
    return maskIntersect;
}

#endif