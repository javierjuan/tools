/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* CImg <-> Python data type conversions                                    *
***************************************************************************/

#ifndef PYCIMG_HPP
#define PYCIMG_HPP

#define cimg_display 0
#include <CImg.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <typeinfo>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace cimg_library;

template<typename T>
using ndarray = py::array_t<T, py::array::c_style | py::array::forcecast>;


class PyCImg
{
public:
    template<typename T>
    static CImg<T> toCImg(const ndarray<T> &input);
    template<typename T>
    static ndarray<T> toNumpy(const CImg<T> &input);
};


template<typename T>
CImg<T> PyCImg::toCImg(const ndarray<T> &input)
{
    py::buffer_info info = input.request();

    if (info.ndim == 0 || (info.ndim == 1 && info.shape[0] == 0))
    {
        return CImg<T>();
    }
        
    if (info.format != py::format_descriptor<T>::format())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible format: expected an <" << typeid(T).name() << "> array" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (info.ndim < 2 || info.ndim > 4)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible buffer dimension: expected a 2D, 3D or 4D array" << std::endl;
        throw std::runtime_error(s.str());
    }

    const int width = info.shape[0];
    const int height = info.shape[1];
    const int depth = info.ndim > 2 ? info.shape[2] : 1;
    const int spectrum = info.ndim > 3 ? info.shape[3] : 1;

    CImg<T> output(width, height, depth, spectrum);
    auto unchecked_input = input.unchecked();
#pragma omp parallel for
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                for (int c = 0; c < spectrum; ++c)
                    output(x, y, z, c) = unchecked_input(x, y, z, c);
            }
        }
    }
    return output;
}


template<typename T>
ndarray<T> PyCImg::toNumpy(const CImg<T> &input)
{
    if (input.is_empty())
    {
        return ndarray<T>();
    }

    const int width = input.width();
    const int height = input.height();
    const int depth = input.depth();
    const int spectrum = input.spectrum();

    ndarray<T> output({width, height, depth, spectrum});
    auto unchecked_output = output.mutable_unchecked();
#pragma omp parallel for
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                for (int c = 0; c < spectrum; ++c)
                    unchecked_output(x, y, z, c) = input(x, y, z, c);
            }
        }
    }
    return output;
}

#endif