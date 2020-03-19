/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Math                                                                     *
***************************************************************************/

#ifndef MATH_HPP
#define MATH_HPP

#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <omp.h>

#define MATH_FLINTMAX 9.0072e+15
#define MATH_EPS 2.220446049250313e-16

#define __STR_FUNCNAME__  std::string(__FUNCTION__)

using namespace Eigen;

template<typename T>
int round(const T value)
{
    return (int) std::ceil(value - 0.5);
}

/***************************** Histogram Edges class *****************************/

class Edges
{
public:
    template<typename T>
    static VectorXd linspace(const T minx, const T maxx, const int N = 100);
    template<typename T>
    static VectorXd binpicker(const T minx, const T maxx, const double rawBinWidth);
    template<class Derived>
    static VectorXd autorule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<class Derived>
    static VectorXd scottsrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<class Derived>
    static VectorXd integerrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<class Derived>
    static VectorXd sturgesrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<class Derived>
    static VectorXd sqrtrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
};

/***************************** Math class *****************************/

class Math
{
public:
    enum Dimension { ROWS, COLUMNS };
    enum KernelType { NORMAL, BOX };

    // Sort, reorder, permute, align
    template<class Derived>
    static typename Derived::PlainObject sort(const DenseBase<Derived> &x, const bool ascending = true, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static typename Derived::PlainObject sort(const DenseBase<Derived> &x, MatrixXi &indices, const bool ascending = true, const Dimension dimension = Math::COLUMNS);
    // Basic statistics
    template<class Derived>
    static VectorXd median(const DenseBase<Derived> &x, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd mode(const DenseBase<Derived> &x, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd var(const DenseBase<Derived> &x, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd std(const DenseBase<Derived> &x, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd mad(const DenseBase<Derived> &x, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd percentile(const DenseBase<Derived> &x, const double p, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd percentile(const DenseBase<Derived> &x, const double p, VectorXi &indices, const Dimension dimension = Math::COLUMNS);
    // Discrete integral and derivatives
    template<class Derived1, class Derived2>
    static VectorXd integral(const DenseBase<Derived1> &x, const DenseBase<Derived2> &y, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static VectorXd integral(const DenseBase<Derived> &y, const double step = 1, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static Matrix<typename Derived::Scalar, Dynamic, Dynamic> diff(const DenseBase<Derived> &x, const Dimension dimension = Math::COLUMNS);
    template<class Derived>
    static MatrixXd gradient(const DenseBase<Derived> &y, const Dimension dimension = Math::COLUMNS);
    template<class Derived1, class Derived2>
    static MatrixXd gradient(const DenseBase<Derived1> &x, const DenseBase<Derived2> &y, const Dimension dimension = Math::COLUMNS);
    // Distribution density estimation
    template<class Derived>
    static std::pair<VectorXi, VectorXd> histogram(const DenseBase<Derived> &x);
    template<class Derived>
    static std::pair<VectorXi, VectorXd> histogram(const DenseBase<Derived> &x, const VectorXd &edges);
    template<class Derived>
    static std::pair<VectorXd, VectorXd> cdf(const DenseBase<Derived> &x);
    template<class Derived>
    static std::pair<VectorXd, VectorXd> cdf(const DenseBase<Derived> &x, const VectorXd &edges);
    // Convolutions
    template<typename T>
    static Matrix<T, Dynamic, 1> conv1d(const Matrix<T, Dynamic, 1> &u, const KernelType vType = Math::NORMAL, const int vSize = 5, const double alpha = 2.5);
    template<typename T1, typename T2>
    static Matrix<T1, Dynamic, 1> conv1d(const Matrix<T1, Dynamic, 1> &u, const Matrix<T2, Dynamic, 1> &v);
};

/***************************** Implementation *****************************/

template<class Derived>
typename Derived::PlainObject _sort(const DenseBase<Derived> &x, const bool ascending)
{
    typename Derived::PlainObject y = x.derived();
    std::sort(y.data(), y.data() + y.size());
    return ascending ? y : y.reverse();
}

template<class Derived>
typename Derived::PlainObject Math::sort(const DenseBase<Derived> &x, const bool ascending, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    typename Derived::PlainObject xsort(x.rows(), x.cols());
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        if (dimension == Math::COLUMNS)
            xsort.col(i) = _sort(x.col(i), ascending);
        else
            xsort.row(i) = _sort(x.row(i), ascending);
    }
    return xsort;
}

template<class Derived>
std::pair<typename Derived::PlainObject, VectorXi> _sorti(const DenseBase<Derived> &x, const bool ascending)
{
    typename Derived::PlainObject y = x.derived();
    VectorXi indices(y.size());
    
#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i)
        indices(i) = i;
    
    std::sort(indices.data(), indices.data() + indices.size(), [&](size_t a, size_t b) { return y(a) < y(b); });
    
#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i)
        y(i) = x(indices(i));
    
    return ascending ? std::make_pair(y, indices) : std::make_pair(y.reverse(), indices.reverse());
}

template<class Derived>
typename Derived::PlainObject Math::sort(const DenseBase<Derived> &x, MatrixXi &indices, const bool ascending, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    indices.resize(x.rows(), x.cols());
    typename Derived::PlainObject xsort(x.rows(), x.cols());
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        if (dimension == Math::COLUMNS)
        {
             std::pair<typename Derived::PlainObject, VectorXi> pair = _sorti(x.col(i), ascending);
             xsort.col(i) = pair.first;
             indices.col(i) = pair.second;
        }
        else
        {
            std::pair<typename Derived::PlainObject, VectorXi> pair = _sorti(x.row(i), ascending);
            xsort.row(i) = pair.first;
            indices.row(i) = pair.second;
        }
    }
    return xsort;
}

template<class Derived>
double _median(const DenseBase<Derived> &x)
{
    typename Derived::PlainObject y = x.derived();
    
    std::sort(y.data(), y.data() + y.size());
    if (((int) y.size() % 2) == 0)
        return (double) (y((int)(y.size() / 2)) + y((int)(y.size() / 2) - 1)) / 2.0;
    else
        return (double)  y((int)(y.size() / 2));
}

template<class Derived>
VectorXd Math::median(const DenseBase<Derived> &x, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    VectorXd medians(N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        medians(i) = dimension == Math::COLUMNS ? _median(x.col(i)) : _median(x.row(i));
    
    return medians;
}

template<class Derived>
double _mode(const DenseBase<Derived> &x)
{
    typename Derived::PlainObject y = x.derived();
    
    std::sort(y.data(), y.data() + y.size());
    
    int counter = 1;
    int max = 0;
    double mode = (double) y(0);
    for (int i = 1; i < (int) y.size(); ++i)
    {
        if (y(i) == y(i - 1))
        {
            counter++;
            if (counter > max)
            {
                max = counter;
                mode = y(i);
            }
        }
        else
        {
            counter = 1;
        }
    }
    return mode;
}

template<class Derived>
VectorXd Math::mode(const DenseBase<Derived> &x, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
        
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    VectorXd modes(N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        modes(i) = dimension == Math::COLUMNS ? _mode(x.col(i)) : _mode(x.row(i));
    
    return modes;
}

template<class Derived>
double _var(const DenseBase<Derived> &x)
{
    typename Derived::PlainObject y = x.derived();
    
    const double mu = (double) y.template cast<double>().mean();
    VectorXd d(x.size());
#pragma omp parallel for
    for (int i = 0; i < x.size(); ++i)
        d(i) = (((double) x(i) - mu) * ((double) x(i) - mu));
    return (double) d.sum() / ((double) x.size() - 1.0);
}

template<class Derived>
VectorXd Math::var(const DenseBase<Derived> &x, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    VectorXd vars(N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        vars(i) = dimension == Math::COLUMNS ? _var(x.col(i)) : _var(x.row(i));
    
    return vars ;
}

template<class Derived>
VectorXd Math::std(const DenseBase<Derived> &x, const Dimension dimension)
{
    return Math::var(x, dimension).cwiseSqrt();
}

template<class Derived>
double _mad(const DenseBase<Derived> &x)
{
    ArrayXd v = x.derived().array();
    return _median(ArrayXd((v - _median(v)).abs()));
}

template<class Derived>
VectorXd Math::mad(const DenseBase<Derived> &x, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    VectorXd mads(N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        mads(i) = dimension == Math::COLUMNS ? _mad(x.col(i)) : _mad(x.row(i));
    
    return mads;
}

template<class Derived>
double _percentile(const DenseBase<Derived> &x, const double p, int &index)
{
    typename Derived::PlainObject y = x.derived();
    
    if (y.size() == 1)
        return (double) y(0);
    
    VectorXi indices(y.size());
    
#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i)
        indices(i) = i;
    
    std::sort(indices.data(), indices.data() + indices.size(), [&](size_t a, size_t b) { return y(a) < y(b); });
    
#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i)
        y(i) = x(indices(i));
    
    if (p == 50)
    {
        index = indices((int) (y.size() / 2));
        if ((y.size() % 2) == 0)
            return (double) (y((int) (y.size() / 2)) + y((int) (y.size() / 2) - 1)) / 2.0;
        else
            return (double)  y((int) (y.size() / 2));
    }
    
    // Get index corresponding to p-percentile
    double r = (p / 100.0) * (double) (y.size() - 1);
    const int k = (int) std::floor(r + 0.5);
    const int kp1 = k + 1 >= (int) y.size() ? (int) y.size() - 1 : k + 1;
    r = r - k;
    
    // Set index
    index = indices(k);
    
    // By interpolation. MATLAB procedure.
    return (double) ((0.5 - r) * (double) y(kp1)) + ((0.5 + r) * (double) y(k));
}

template<class Derived>
VectorXd Math::percentile(const DenseBase<Derived> &x, const double p, const Dimension dimension)
{
    VectorXi indices;
    return Math::percentile(x, p, indices, dimension);
}

template<class Derived>
VectorXd Math::percentile(const DenseBase<Derived> &x, const double p, VectorXi &indices, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    VectorXd percentiles(N);
    indices.resize(N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int index = 0;
        percentiles(i) = dimension == Math::COLUMNS ? _percentile(x.col(i), p, index) : _percentile(x.row(i), p, index);
        indices(i) = index;
    }
    
    return percentiles;
}

template<class Derived1, class Derived2>
double _integral(const DenseBase<Derived1> &x, const DenseBase<Derived2> &y)
{
    typename Derived1::PlainObject xd = x.derived();
    typename Derived2::PlainObject yd = y.derived();
    
    double result = 0;
    for (int i = 0; i < (int) xd.size() - 1; ++i)
        result += (double) (xd(i+1) - xd(i)) * (yd(i+1) + yd(i));
    return result / 2.0;
}

template<class Derived1, class Derived2>
VectorXd Math::integral(const DenseBase<Derived1> &x, const DenseBase<Derived2> &y, const Dimension dimension)
{
    assert(x.rows() == y.rows());
    assert((x.cols() == y.cols()) || (x.cols() == 1));
    
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    VectorXd integrals(N);
    if (x.cols() == y.cols())
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            integrals(i) = dimension == Math::COLUMNS ? _integral(x.col(i), y.col(i)) : _integral(x.col(i), y.col(i));
    }
    if (x.cols() == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            integrals(i) = dimension == Math::COLUMNS ? _integral(x, y.col(i)) : _integral(x, y.col(i));
    }
    return integrals;
}

template<class Derived>
double _integral(const DenseBase<Derived> &y, const double step)
{
    typename Derived::PlainObject yd = y.derived();
    
    double result = 0;
    for (int i = 0; i < (int) yd.size() - 1; ++i)
        result += (double) step * (yd(i+1) + yd(i));
    return result / 2.0;
}

template<class Derived>
VectorXd Math::integral(const DenseBase<Derived> &y, const double step, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? y.cols() : y.rows();
    
    VectorXd integrals(N);
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            integrals(i) = dimension == Math::COLUMNS ? _integral(y.col(i), step) : _integral(y.row(i), step);
    
    return integrals;
}

template<class Derived>
Matrix<typename Derived::Scalar, Dynamic, 1> _diff(const DenseBase<Derived> &x)
{
    typename Derived::PlainObject y = x.derived();
    
    Matrix<typename Derived::Scalar, Dynamic, 1> d(y.size() - 1);
    for (int i = 1, j = 0; i < (int) y.size(); ++i, ++j)
        d(j) = y(i) - y(j);
    return d;
}

template<class Derived>
Matrix<typename Derived::Scalar, Dynamic, Dynamic> Math::diff(const DenseBase<Derived> &x, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? x.cols() : x.rows();
    
    Matrix<typename Derived::Scalar, Dynamic, Dynamic> diffs(x.rows() - 1, x.cols());
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            if (dimension == Math::COLUMNS)
                diffs.col(i) = _diff(x.col(i));
            else
                diffs.row(i) = _diff(x.row(i));
        }
    
    return diffs;
}

template<class Derived>
VectorXd _gradient(const DenseBase<Derived> &y)
{
    typename Derived::PlainObject yd = y.derived();
    
    const int N = (int) yd.size();
    if (N == 1)
        return yd.derived().template cast<double>();
    
    VectorXd g(yd.size());
    // Forward differences on left and right edges
    g(0) = yd(1) - yd(0);
    g(N - 1) = yd(N - 1) - yd(N - 2);
    // Centered differences on interior points
    if (N > 2)
    {
#pragma omp parallel for
        for (int i = 1; i < N - 1; ++i)
            g(i) = (double) (yd(i+1) - yd(i-1)) / 2.0;
    }
    return g;
}

template<class Derived>
MatrixXd Math::gradient(const DenseBase<Derived> &y, const Dimension dimension)
{
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? y.cols() : y.rows();
    
    MatrixXd gradients(y.rows(), y.cols());
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            if (dimension == Math::COLUMNS)
                gradients.col(i) = _gradient(y.col(i));
            else
                gradients.row(i) = _gradient(y.row(i));
        }
    
    return gradients;
}

template<class Derived1, class Derived2>
VectorXd _gradient(const DenseBase<Derived1> &x, const DenseBase<Derived2> &y)
{
    typename Derived1::PlainObject xd = x.derived();
    typename Derived1::PlainObject yd = y.derived();
    
    const int N = (int) yd.size();
    if (N == 1)
        return yd.derived().template cast<double>();
    
    VectorXd g(yd.size());
    // Forward differences on left and right edges
    g(0) = (yd(1) - yd(0)) / (xd(1) - xd(0));
    g(N - 1) = (yd(N - 1) - yd(N - 2)) / (xd(N - 1) - xd(N - 2));
    // Centered differences on interior points
    if (N > 2)
    {
#pragma omp parallel for
        for (int i = 1; i < N - 1; ++i)
            g(i) = (double) (yd(i+1) - yd(i-1)) / (xd(i+1) - xd(i-1));
    }
    return g;
}

template<class Derived1, class Derived2>
MatrixXd Math::gradient(const DenseBase<Derived1> &x, const DenseBase<Derived2> &y, const Dimension dimension)
{
    assert(x.rows() == y.rows());
    assert((x.cols() == y.cols()) || (x.cols() == 1));
    
    if (dimension != COLUMNS && dimension != ROWS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Unsupported <dimension> parameter value." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int N = dimension == Math::COLUMNS ? y.cols() : y.rows();
    
    MatrixXd gradients(y.rows(), y.cols());
    if (x.cols() == y.cols())
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            if (dimension == Math::COLUMNS)
                gradients.col(i) = _gradient(x.col(i), y.col(i));
            else
                gradients.row(i) = _gradient(x.row(i), y.row(i));
        }
    }
    if (x.cols() == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            if (dimension == Math::COLUMNS)
                gradients.col(i) = _gradient(x, y.col(i));
            else
                gradients.row(i) = _gradient(x, y.row(i));
        }
    }
    
    return gradients;
}

template<class Derived>
std::pair<VectorXi, VectorXd> Math::histogram(const DenseBase<Derived> &x)
{
    assert((x.rows() == 1) || (x.cols() == 1));
        
    typename Derived::PlainObject y = x.derived();
    
    const VectorXd edges = Edges::autorule(y, y.minCoeff(), y.maxCoeff(), false);
    
    return Math::histogram(y, edges);
}

template<class Derived>
std::pair<VectorXi, VectorXd> Math::histogram(const DenseBase<Derived> &x, const VectorXd &edges)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    assert((edges.rows() == 1) || (edges.cols() == 1));
    
    typename Derived::PlainObject y = x.derived();
    
    const double binwidth = edges(1) - edges(0);
    VectorXi hist = VectorXi::Zero(edges.size());
    for (int i = 0; i < y.size(); ++i)
    {
        const int pos = (int) std::ceil(((y(i) - edges(0)) / binwidth));  // Check this method. Histograms are not exactly the same than MATLAB
        hist(pos)++;
    }
    return std::make_pair(hist, edges);
}

template<class Derived>
std::pair<VectorXd, VectorXd> Math::cdf(const DenseBase<Derived> &x)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    
    typename Derived::PlainObject y = x.derived();
    
    const VectorXd edges = Edges::autorule(y, y.minCoeff(), y.maxCoeff(), false);
    
    return Math::cdf(y, edges);
}

template<class Derived>
std::pair<VectorXd, VectorXd> Math::cdf(const DenseBase<Derived> &x, const VectorXd &edges)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    assert((edges.rows() == 1) || (edges.cols() == 1));
    
    typename Derived::PlainObject y = x.derived();
    
    const std::pair<VectorXi, VectorXd> H = Math::histogram(y, edges);
    const VectorXi hist = H.first;
    
    VectorXd cum = VectorXd::Zero(hist.size());
    cum(0) = hist(0);
    for (int i = 1; i < cum.size(); ++i)
    {
        cum(i) = (double) (cum(i-1) + hist(i));
    }

    const double sum = (double) hist.sum();
#pragma omp parallel for
    for (int i = 0; i < cum.size(); ++i)
    {
        cum(i) /= sum;
    }

    return std::make_pair(cum, edges);
}

template<typename T>
Matrix<T, Dynamic, 1> Math::conv1d(const Matrix<T, Dynamic, 1> &u, const KernelType vType, const int vSize, const double alpha)
{
    VectorXd v(vSize);
    
    const int vOffset = (int) (vSize / 2);
    switch (vType)
    {
    case Math::NORMAL:
        {
        const double N = (double) (vSize - 1) / 2.0;
        for (int i = 0, j = -vOffset; i < vSize; ++i, ++j)
            v(i) = std::exp(-0.5 * ((alpha * j) / N) * ((alpha * j) / N));
        v = v / v.sum();
        }
        break;
    case Math::BOX:
        {
        for (int i = 0; i < vSize; ++i)
            v(i) = 1.0 / vSize;
        }
        break;
    default:
        {
        const double N = ((double) vSize - 1.0) / 2.0;
        for (int i = 0, j = -vOffset; i < vSize; ++i, ++j)
            v(i) = std::exp(-0.5 * ((alpha * j) / N) * ((alpha * j) / N));
        v = v / v.sum();
        }
        break;
    }
    return Math::conv1d(u, v);
}

template<typename T1, typename T2>
Matrix<T1, Dynamic, 1> Math::conv1d(const Matrix<T1, Dynamic, 1> &u, const Matrix<T2, Dynamic, 1> &v)
{
    assert((v.size() % 2) != 0);
    
    if (v.sum() < 1.0 - MATH_EPS || v.sum() > 1.0 + MATH_EPS)
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << "==> Kernel v does not sum 1." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int vSize = (int) v.size();
    const int vOffset = (int) (vSize / 2);
    
    VectorXd um(u.size() + (vOffset * 2));
#pragma omp parallel for
    for (int i = 0; i < (int) u.size(); ++i)
        um(i + vOffset) = u(i);
    for (int i = 0, j = vOffset; i < vOffset; ++i, --j)
        um(i) = u(j);
    for (int i = (int) um.size() - 1, j = (int) u.size() - vOffset - 1; i >= (int) um.size() - vOffset; --i, ++j)
        um(i) = u(j);

    VectorXd w(u.size());
#pragma omp parallel for
    for (int i = 0; i < w.size(); ++i)
        w(i) = um.segment(i, vSize).transpose() * v;
    
    return w;
}

/***************************** Implementation of Histogram Edges class *****************************/
/***************************** must be here with the declaration first *****************************/

template<typename T>
VectorXd Edges::linspace(const T minx, const T maxx, const int N)
{
    VectorXd y(N);
    
    const int N1 = N - 1;
    const T c = (maxx - minx) * (T)(N1 - 1);
    if (!_finite(c))
    {
        if (!_finite(maxx - minx))
        {
            for (int i = 0; i < N; ++i)
                y(i) = (double) minx + ((double) maxx / (double) N1) * i - ((double) minx / (double) N1) * i;
        }
        else
        {
            for (int i = 0; i < N; ++i)
                y(i) = (double) minx + i * ((double) (maxx - minx) / (double) N1);
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
            y(i) = (double) minx + i * ((double) (maxx - minx) / (double) N1);
    }
    if (y.size() > 0 )
    {
        y(0) = (double) minx;
        y(N1) = (double) maxx;
    }
    return y;
}

template<typename T>
VectorXd Edges::binpicker(const T minx, const T maxx, const double rawBinWidth)
{
    const double xscale = (double) std::max(std::abs((double) minx), std::abs((double) maxx));
    const double xrange = (double) (maxx - minx);
    
    const double powOfTen = std::pow(10.0, std::floor(std::log10(rawBinWidth)));
    const double relSize = rawBinWidth / powOfTen;
    double binWidth;
    if (relSize < 1.5)
        binWidth = 1 * powOfTen;
    else if (relSize < 2.5)
        binWidth = 2 * powOfTen;
    else if (relSize < 4)
        binWidth = 3 * powOfTen;
    else if (relSize < 7.5)
        binWidth = 5 * powOfTen;
    else
        binWidth = 10 * powOfTen;
    
    const double leftEdge = std::min(binWidth * std::floor((double) minx / binWidth), (double) minx);
    const int nbinsActual = std::max(1, (int) std::ceil((double) (maxx - leftEdge) / binWidth));
    const double rightEdge = std::max(leftEdge + nbinsActual * binWidth, (double) maxx);
    
    VectorXd y(nbinsActual + 1);
    for (int i = 0; i < nbinsActual; ++i)
        y(i) = leftEdge + ((double) i * binWidth);
    y(nbinsActual) = rightEdge;
    return y;
}

template<class Derived>
VectorXd Edges::autorule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    
    typename Derived::PlainObject y = x.derived();
    
    const typename Derived::Scalar xrange = maxx - minx;
    if (y.size() > 0 && std::is_integral<typename Derived::Scalar>::value && xrange <= 50 && maxx <= (MATH_FLINTMAX / 2.0) && minx >= -(MATH_FLINTMAX / 2.0))
        return Edges::integerrule(y, minx, maxx, hardlimits);
    else
        return Edges::scottsrule(y, minx, maxx, hardlimits);
}

template<class Derived>
VectorXd Edges::scottsrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    
    typename Derived::PlainObject y = x.derived();
    
    const VectorXd sigma = Math::std(y);
    const double binwidth = 3.5 * sigma(0) / std::pow((double) y.size(), 1.0 / 3.0);
    if (hardlimits)
    {
        const int nbins = (int) std::ceil((double) (maxx - minx) / binwidth);
        return Edges::linspace(minx, maxx, nbins);
    }
    else
    {
        return Edges::binpicker(minx, maxx, binwidth);
    }
}

template<class Derived>
VectorXd Edges::integerrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    
    if (maxx > (MATH_FLINTMAX / 2.0) || minx < -(MATH_FLINTMAX / 2.0))
    {
        std::stringstream s;
        s << __STR_FUNCNAME__ << " ==> Input out of integer range." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    typename Derived::PlainObject y = x.derived();
    
    const int maximumBins = 65536;
    const typename Derived::Scalar xscale = y.array().abs().maxCoeff();
    const typename Derived::Scalar xrange = y.maxCoeff() - y.minCoeff();
    
    double binwidth = 1;
    if (xrange > maximumBins)
        binwidth = std::pow(10.0, std::ceil(std::log10((double) xrange / (double) maximumBins)));
    
    if (hardlimits)
    {
        const double minxi = binwidth * std::ceil((double) minx / binwidth)  + 0.5;
        const double maxxi = binwidth * std::floor((double) maxx / binwidth) - 0.5;
        const int N = (int) std::floor(((maxxi - minxi) / binwidth) + 3);
        VectorXd y(N);
        int j = 1;
        for (double i = minxi; i <= maxxi; i += binwidth)
        {
            y(j) = i;
            j++;
        }
        y(0) = minx;
        y(N-1) = maxx;
        return y;
    }
    else
    {
        const double minxi = std::floor(binwidth * round((double) minx / binwidth)) - 0.5 * binwidth;
        const double maxxi = std::ceil(binwidth * round((double) maxx / binwidth)) + 0.5 * binwidth;
        const int N = (int) std::floor(((maxxi - minxi) / binwidth) + 1);
        VectorXd y(N);
        int j = 0;
        for (double i = minxi; i <= maxxi; i += binwidth)
        {
            y(j) = i;
            j++;
        }
        return y;
    }
}

template<class Derived>
VectorXd Edges::sturgesrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    
    const int nbins = std::max(std::ceil(std::log2(x.size()) + 1), 1);
    if (hardlimits)
    {
        return Edges::linspace(minx, maxx, nbins);
    }
    else
    {
        return Edges::binpicker(minx, maxx, (double)(maxx - minx) / (double) nbins);
    }
}

template<class Derived>
VectorXd Edges::sqrtrule(const DenseBase<Derived> &x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    assert((x.rows() == 1) || (x.cols() == 1));
    
    const int nbins = std::max(std::ceil(std::sqrt(x.size())), 1);
    if (hardlimits)
    {
        return Edges::linspace(minx, maxx, nbins);
    }
    else
    {
        return Edges::binpicker(minx, maxx, (double)(maxx - minx) / (double) nbins);
    }
}

#endif