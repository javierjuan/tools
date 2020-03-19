/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2018 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Eigen <-> CImg data type conversions                                     *
***************************************************************************/

#ifndef EIGENCIMG_HPP
#define EIGENCIMG_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <stdexcept>

#define __STR_FUNCNAME__  std::string(__FUNCTION__)

using namespace cimg_library;
using namespace Eigen;

class EigenCImg
{
public:
    template <typename T>
    static Matrix<T, Dynamic, Dynamic> toEigen(const CImg<T> &image);
    template <typename T>
    static Matrix<T, Dynamic, Dynamic> toEigen(const CImg<T> &image, const CImg<bool> &mask);
    template <class Derived>
    static CImg<typename Derived::Scalar> toCImg(const DenseBase<Derived> &data, const CImg<bool> &mask);
    template<typename T, int Dimensions>
    static CImg<T> toCImg(const Tensor<T, Dimensions> &data, const CImg<bool> &mask);
};


template <typename T>
Matrix<T, Dynamic, Dynamic> EigenCImg::toEigen(const CImg<T> &image)
{
    int i = 0;
    Matrix<T, Dynamic, Dynamic> data(image.width() * image.height() * image.depth(), image.spectrum());
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                for (int c = 0; c < image.spectrum(); ++c)
                    data(i, c) = (T) image(x, y, z, c);
                ++i;
            }
        }
    }
    return data;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> EigenCImg::toEigen(const CImg<T> &image, const CImg<bool> &mask)
{
    if (image.height() != mask.height() || image.width() != mask.width() || image.depth() != mask.depth())
    {
        std::stringstream s;
        s << "In function " << __STR_FUNCNAME__ << " ==> Image and mask dimensions must agree." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    int i = 0;
    Matrix<T, Dynamic, Dynamic> data(mask.size(), image.spectrum());
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                if (!mask(x, y, z))
                    continue;
                
                for (int c = 0; c < image.spectrum(); ++c)
                    data(i, c) = (T) image(x, y, z, c);
                ++i;
            }
        }
    }
    data.conservativeResize(i, data.cols());
    return data;
}

template <class Derived>
CImg<typename Derived::Scalar> EigenCImg::toCImg(const DenseBase<Derived> &data, const CImg<bool> &mask)
{
    if (mask.sum() != data.rows())
    {
        std::stringstream s;
        s << "In function " << __STR_FUNCNAME__ << " ==> Number of positive mask elements must agree with data rows." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    int i = 0;
    CImg<typename Derived::Scalar> image(mask.width(), mask.height(), mask.depth(), data.cols());
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                if (!mask(x, y, z))
                {
                    for (int c = 0; c < data.cols(); ++c)
                        image(x, y, z, c) = (typename Derived::Scalar) 0;
                }
                else
                {
                    for (int c = 0; c < data.cols(); ++c)
                        image(x, y, z, c) = (typename Derived::Scalar) data(i, c);
                    ++i;
                }
            }
        }
    }
    return image;
}

template<typename T, int Dimensions>
CImg<T> EigenCImg::toCImg(const Tensor<T, Dimensions> &data, const CImg<bool> &mask)
{
    if (mask.sum() != (int) data.dimension(0))
    {
        std::stringstream s;
        s << "In function " << __STR_FUNCNAME__ << " ==> Number of positive mask elements must agree with data rows." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int K = data.dimension(Dimensions - 1);
    
    typename Eigen::Tensor<T, Dimensions-2>::Dimensions dims;
    for (int i = 0; i < Dimensions - 2; ++i)
        dims[i] = i + 1;
    
    Eigen::Tensor<T, 2> dataReduced = data.sum(dims);
    
    int i = 0;
    CImg<T> image(mask.width(), mask.height(), mask.depth(), K);
    for (int x = 0; x < image.width(); ++x)
    {
        for (int y = 0; y < image.height(); ++y)
        {
            for (int z = 0; z < image.depth(); ++z)
            {
                if (!mask(x, y, z))
                {
                    for (int c = 0; c < K; ++c)
                        image(x, y, z, c) = (T) 0;
                }
                else
                {
                    for (int c = 0; c < K; ++c)
                        image(x, y, z, c) = (T) dataReduced(i, c);
                    ++i;
                }
            }
        }
    }
    return image;
}

#endif