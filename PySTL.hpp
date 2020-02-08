/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* STL <-> Python data type conversions                                     *
***************************************************************************/

#ifndef PYSTL_HPP
#define PYSTL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <typeinfo>
#include <stdexcept>
#include <string>

namespace py = pybind11;

template<typename T>
using ndarray = py::array_t<T, py::array::c_style | py::array::forcecast>;


class PySTL
{
public:
    template<typename T>
    static std::vector<T> toVector(const ndarray<T> &input);
    template<typename T>
    static ndarray<T> toNumpy(const std::vector<T> &input);
};


template<typename T>
std::vector<T> PySTL::toVector(const ndarray<T> &input)
{
    py::buffer_info info = input.request();

    if (info.ndim == 1 and info.shape[0] == 0)
    {
        return std::vector<T>();
    }

    if (info.ndim > 1)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible format: expected a 1D numpy array" << std::endl;
        throw std::runtime_error(s.str());
    }
        
    if (info.format != py::format_descriptor<T>::format())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible format: expected an <" << typeid(T).name() << "> array" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    std::vector<T> output(info.shape[0]);
    const T *pyPtr = (const T*) info.ptr;

    #pragma omp parallel for
    for (int i = 0; i < (int) info.shape[0]; ++i)
    {
        output[i] = static_cast<T>(pyPtr[i]);
    }
    return output;
}


template<typename T>
ndarray<T> PySTL::toNumpy(const std::vector<T> &input)
{
    if (input.empty())
    {
        return ndarray<T>();
    }

    ndarray<T> output({input.size()});
    py::buffer_info info = output.request();

    T *pyPtr = (T*) info.ptr;

    #pragma omp parallel for
    for (int i = 0; i < (int) input.size(); ++i)
    {
        pyPtr[i] = static_cast<T>(input[i]);
    }

    return output;
}

#endif