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
#include <tuple>
#include <omp.h>

#define MATH_FLINTMAX 9.0072e+15
#define MATH_EPS 2.220446049250313e-16
#define MATH_INF 1e300

#if _WIN32
#define __PRETTY_FUNCTION__  std::string(__FUNCTION__)
#endif

using namespace Eigen;

template<typename T>
int round(const T value)
{
    return (int) std::ceil(value - 0.5);
}

template<typename T>
int sign(const T value)
{
    return value == 0 ? 0 : value > 0 ? 1 : -1;
}

/***************************** Histogram Edges class *****************************/

class Edges
{
public:
    template<typename T>
    static VectorXd binpicker(const T minx, const T maxx, const double rawBinWidth);
    template<typename Derived>
    static VectorXd autorule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<typename Derived>
    static VectorXd scottsrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<typename Derived>
    static VectorXd integerrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<typename Derived>
    static VectorXd sturgesrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
    template<typename Derived>
    static VectorXd sqrtrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits);
};

/***************************** Math class *****************************/

class Math
{
public:
    enum Order { ASCENDING, DESCENDING };
    enum Interpolation { PCHIP, MAKIMA };
    enum Kernel { NORMAL, BOX };

    template<typename T>
    static bool isClose(const T x, const T y, const double rtol = 1.e-5, const double atol = 1.e-8);
    // Sort, reorder, permute, align
    template<typename Derived>
    static typename Derived::PlainObject sort(const DenseBase<Derived>& x, const Order order = Math::ASCENDING);
    template<typename Derived1, typename Derived2>
    static typename Derived1::PlainObject sort(const DenseBase<Derived1>& x, DenseBase<Derived2> &indices, const Order order = Math::ASCENDING);
    // Basic statistics
    template<typename Derived>
    static double mean(const DenseBase<Derived>& x);
    template<typename Derived>
    static double median(const DenseBase<Derived>& x);
    template<typename Derived>
    static typename Derived::Scalar mode(const DenseBase<Derived>& x);
    template<typename Derived>
    static double var(const DenseBase<Derived>& x);
    template<typename Derived>
    static double std(const DenseBase<Derived>& x);
    template<typename Derived>
    static double mad(const DenseBase<Derived>& x);
    template<typename Derived>
    static double percentile(const DenseBase<Derived>& x, const double p);
    template<typename Derived>
    static double percentile(const DenseBase<Derived>& x, const double p, int& index);
    // Linspace
    template<typename T>
    static VectorXd linspace(const T minx, const T maxx, const int N = 100);
    // Discrete integral and derivatives
    template<typename Derived1, typename Derived2>
    static double integral(const DenseBase<Derived1>& x, const DenseBase<Derived2>& y);
    template<typename Derived>
    static double integral(const DenseBase<Derived>& y, const double step = 1);
    template<typename Derived>
    static Matrix<typename Derived::Scalar, Dynamic, 1> diff(const DenseBase<Derived>& x);
    template<typename Derived>
    static VectorXd gradient(const DenseBase<Derived>& y);
    template<typename Derived1, typename Derived2>
    static VectorXd gradient(const DenseBase<Derived1>& x, const DenseBase<Derived2>& y);
    // Distribution density estimation
    template<typename Derived>
    static std::tuple<VectorXi, VectorXd, VectorXi> histogram(const DenseBase<Derived>& x);
    template<typename Derived>
    static std::tuple<VectorXi, VectorXd, VectorXi> histogram(const DenseBase<Derived>& x, const VectorXd& edges);
    template<typename Derived>
    static std::tuple<VectorXd, VectorXd, VectorXi> cdf(const DenseBase<Derived>& x);
    template<typename Derived>
    static std::tuple<VectorXd, VectorXd, VectorXi> cdf(const DenseBase<Derived>& x, const VectorXd& edges);
    // Convolutions
    template<typename Derived>
    static VectorXd conv1d(const DenseBase<Derived>& x, const Kernel type = Math::NORMAL, const int size = 5, const double alpha = 2.5);
    template<typename Derived>
    static VectorXd conv1d(const DenseBase<Derived>& x, const VectorXd &k);
    // Interpolation
    template<typename Derived1, typename Derived2, typename Derived3>
    static VectorXd interp1(const DenseBase<Derived1>& x, const DenseBase<Derived2>& y, const DenseBase<Derived3> &xq, const Interpolation type = Math::MAKIMA);
};

/***************************** Implementation *****************************/
template<typename T>
bool Math::isClose(const T x, const T y, const double rtol, const double atol)
{
    return std::abs(x - y) <= (atol + rtol * std::abs(y));
}

template<typename Derived>
typename Derived::PlainObject Math::sort(const DenseBase<Derived>& x, const Order order)
{
    if (order != ASCENDING && order != DESCENDING)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << ": Unsupported <order> parameter." << std::endl;
        throw std::runtime_error(s.str());
    }

    typename Derived::PlainObject y = x.derived();

    std::sort(y.data(), y.data() + y.size());
    return order == Math::ASCENDING ? y : y.reverse();
}

template<typename Derived1, typename Derived2>
typename Derived1::PlainObject Math::sort(const DenseBase<Derived1>& x, DenseBase<Derived2> &indices, const Order order)
{
    if (order != ASCENDING && order != DESCENDING)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << ": Unsupported <order> parameter." << std::endl;
        throw std::runtime_error(s.str());
    }

    typename Derived1::PlainObject y = x.derived();
    typename Derived2::PlainObject& z = indices.derived();

    z.resize(y.rows(), y.cols());
#pragma omp parallel for
    for (Index i = 0; i < z.size(); ++i)
        z(i) = i;
    
    std::sort(z.data(), z.data() + z.size(), [&](size_t a, size_t b) { return y(a) < y(b); });
    
#pragma omp parallel for
    for (Index i = 0; i < z.size(); ++i)
        y(i) = x((Index) z(i));
    
    return order == Math::ASCENDING ? y : y.reverse();
}

template<typename Derived>
double Math::mean(const DenseBase<Derived>& x)
{
    const typename Derived::PlainObject& y = x.derived();
    return (double) y.template cast<double>().mean();
}

template<typename Derived>
double Math::median(const DenseBase<Derived>& x)
{
    typename Derived::PlainObject y = x.derived();

    std::sort(y.data(), y.data() + y.size());
    if (((int) y.size() % 2) == 0)
        return (double) (y((Index)(y.size() / 2)) + y((Index)(y.size() / 2) - 1)) / 2.0;
    else
        return (double) y((Index)(y.size() / 2));
}

template<typename Derived>
typename Derived::Scalar Math::mode(const DenseBase<Derived>& x)
{
    typename Derived::PlainObject y = x.derived();
    
    std::sort(y.data(), y.data() + y.size());
    
    int max = 1;
    int counter = 1;
    typename Derived::Scalar mode = y(0);
    for (Index i = 1; i < y.size(); ++i)
    {
        if (y(i) == y(i - 1))
        {
            ++counter;
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

template<typename Derived>
double Math::var(const DenseBase<Derived>& x)
{
    const double mu = Math::mean(x);
    double diff = 0;
    for (Index i = 0; i < x.size(); ++i)
        diff += (((double) x(i) - mu) * ((double) x(i) - mu));
    return diff / ((double) x.size() - 1.0);
}

template<typename Derived>
double Math::std(const DenseBase<Derived>& x)
{
    return std::sqrt(Math::var(x));
}

template<typename Derived>
double Math::mad(const DenseBase<Derived>& x)
{
    const typename Derived::PlainObject& y = x.derived();
    return Math::median((y - Math::median(y)).abs());
}

template<typename Derived>
double Math::percentile(const DenseBase<Derived>& x, const double p)
{
    int index = 0;
    return Math::percentile(x, p, index);
}

template<typename Derived>
double Math::percentile(const DenseBase<Derived>& x, const double p, int& index)
{
    if (p < 0 || p > 100)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << ": Percentile must be in the range [0, 100]" << std::endl;
        throw std::runtime_error(s.str());
    }

    if (x.size() == 1)
        return (double) x(0);

    VectorXi indices;
    const typename Derived::PlainObject& y = Math::sort(x, indices);
    
    if (p == 50)
    {
        index = indices((int) (y.size() / 2));
        if (((int) y.size() % 2) == 0)
            return (double) (y((Index) (y.size() / 2)) + y((Index)(y.size() / 2) - 1)) / 2.0;
        else
            return (double) y((Index) (y.size() / 2));
    }
    
    // Get index corresponding to p-percentile
    double r = (p / 100.0) * (double) (y.size() - 1);
    const int k = (int) std::floor(r);
    const int kp1 = k + 1 >= (int) y.size() ? (int) y.size() - 1 : k + 1;
    r = r - k;

    // Set index
    index = indices(k);

    // By interpolation
    return r == 0 ? (double) y(k) : (r * (double) y(kp1)) + ((1.0 - r) * (double) y(k));
}

template<typename T>
VectorXd Math::linspace(const T minx, const T maxx, const int N)
{
    if (N < 3)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << ": Number of space points must be greater or equal than 3" << std::endl;
        throw std::runtime_error(s.str());
    }

    VectorXd y(N);
    
    const int N1 = N - 1;
    const T c = (maxx - minx) * (T)(N1 - 1);
    if (!std::isfinite(c))
    {
        if (!std::isfinite(maxx - minx))
        {
            for (Index i = 0; i < N; ++i)
                y(i) = (double) minx + ((double) maxx / (double) N1) * i - ((double) minx / (double) N1) * i;
        }
        else
        {
            for (Index i = 0; i < N; ++i)
                y(i) = (double) minx + i * ((double) (maxx - minx) / (double) N1);
        }
    }
    else
    {
        for (Index i = 0; i < N; ++i)
            y(i) = (double) minx + i * ((double) (maxx - minx) / (double) N1);
    }
    if (y.size() > 0 )
    {
        y(0) = (double) minx;
        y(N1) = (double) maxx;
    }
    return y;
}

template<typename Derived1, typename Derived2>
double Math::integral(const DenseBase<Derived1>& x, const DenseBase<Derived2>& y)
{
    if (x.size() != y.size())
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x and y must be of the same size." << std::endl;
        throw std::runtime_error(s.str());
    }

    double result = 0;
    for (Index i = 0; i < y.size() - 1; ++i)
        result += (double) (x(i+1) - x(i)) * (y(i+1) + y(i));
    return result / 2.0;
}

template<typename Derived>
double Math::integral(const DenseBase<Derived>& y, const double step)
{
    double result = 0;
    for (Index i = 0; i < y.size() - 1; ++i)
        result += (double) step * (y(i+1) + y(i));
    return result / 2.0;
}

template<typename Derived>
Matrix<typename Derived::Scalar, Dynamic, 1> Math::diff(const DenseBase<Derived>& x)
{
    if (x.rows() != 1 && x.cols() != 1)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x must be a row or column vector/array." << std::endl;
        throw std::runtime_error(s.str());
    }

    Matrix<typename Derived::Scalar, Dynamic, 1> d(x.size() - 1);
#pragma omp parallel for
    for (Index i = 0; i < x.size() - 1; ++i)
        d(i) = x(i+1) - x(i);
    return d;
}

template<typename Derived>
VectorXd Math::gradient(const DenseBase<Derived> &y)
{
    if (y.rows() != 1 && y.cols() != 1)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x must be row or column vector/array." << std::endl;
        throw std::runtime_error(s.str());
    }

    const int N = y.size();
    if (N == 1)
        return y.derived().template cast<double>();
    
    VectorXd g(y.size());
    // Forward differences on left and right edges
    g(0) = y(1) - y(0);
    g(N - 1) = y(N - 1) - y(N - 2);
    // Centered differences on interior points
    if (N > 2)
    {
#pragma omp parallel for
        for (Index i = 1; i < N - 1; ++i)
            g(i) = (double) (y(i+1) - y(i-1)) / 2.0;
    }
    return g;
}

template<typename Derived1, typename Derived2>
VectorXd Math::gradient(const DenseBase<Derived1>& x, const DenseBase<Derived2>& y)
{
    if (x.size() != y.size())
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x and y must be of the same size." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (y.rows() != 1 && y.cols() != 1)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x and y must be row or column vectors/arrays." << std::endl;
        throw std::runtime_error(s.str());
    }

    const int N = y.size();
    if (N == 1)
        return y.derived().template cast<double>();
    
    VectorXd g(y.size());
    // Forward differences on left and right edges
    g(0) = (y(1) - y(0)) / (x(1) - x(0));
    g(N - 1) = (y(N - 1) - y(N - 2)) / (x(N - 1) - x(N - 2));
    // Centered differences on interior points
    if (N > 2)
    {
#pragma omp parallel for
        for (Index i = 1; i < N - 1; ++i)
            g(i) = (double) (y(i+1) - y(i-1)) / (x(i+1) - x(i-1));
    }
    return g;
}

template<typename Derived>
std::tuple<VectorXi, VectorXd, VectorXi> Math::histogram(const DenseBase<Derived>& x)
{
    return Math::histogram(x, Edges::autorule(x, x.minCoeff(), x.maxCoeff(), false));
}

template<typename Derived>
std::tuple<VectorXi, VectorXd, VectorXi> Math::histogram(const DenseBase<Derived>& x, const VectorXd& edges)
{
    if (x.size() < 3)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x must have at least more than 3 elements." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!Math::isClose(std::abs(edges(1) - edges(0)), std::abs(edges(2) - edges(1))) ||
        !Math::isClose(std::abs(edges(2) - edges(1)), std::abs(edges(3) - edges(2))))
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> Unsupported binwidth. <histogram> function only prepared to deal with equal binwidths" << std::endl;
        throw std::runtime_error(s.str());
    }

    const int numEqualBins = (int) edges.size() - 1;
    const double firstEdge = edges(0);
    const double lastEdge = edges(numEqualBins);
    const double norm = (double) numEqualBins / std::abs(lastEdge - firstEdge);
    VectorXi hist = VectorXi::Zero(numEqualBins);
    VectorXi bin(x.size());
    for (Index i = 0; i < x.size(); ++i)
    {
        if (x(i) < firstEdge || x(i) > lastEdge)
            continue;

        int pos = (int) (std::abs(x(i) - firstEdge) * norm);
        pos = pos == numEqualBins ? pos - 1 : pos;
        pos = x(i) < edges(pos) ? pos - 1 : pos;
        pos = (x(i) >= edges(pos + 1)) && (pos != (numEqualBins - 1)) ? pos + 1 : pos;
        hist(pos)++;
        bin(i) = pos;
    }

    return std::make_tuple(hist, edges, bin);
}

template<typename Derived>
std::tuple<VectorXd, VectorXd, VectorXi> Math::cdf(const DenseBase<Derived>& x)
{
    return Math::cdf(x, Edges::autorule(x, x.minCoeff(), x.maxCoeff(), false));
}

template<typename Derived>
std::tuple<VectorXd, VectorXd, VectorXi> Math::cdf(const DenseBase<Derived>& x, const VectorXd& edges)
{
    const auto H = Math::histogram(x, edges);
    const VectorXi hist = std::get<0>(H);
    const VectorXi bin = std::get<2>(H);
    
    VectorXd cum = VectorXd::Zero(hist.size());
    cum(0) = hist(0);
    for (Index i = 1; i < cum.size(); ++i)
    {
        cum(i) = (double) (cum(i-1) + hist(i));
    }

    const double sum = (double) hist.sum();
#pragma omp parallel for
    for (Index i = 0; i < cum.size(); ++i)
    {
        cum(i) /= sum;
    }

    return std::make_tuple(cum, edges, bin);
}

template<typename Derived>
VectorXd Math::conv1d(const DenseBase<Derived>& x, const Kernel type, const int size, const double alpha)
{
    VectorXd k(size);
    
    const int vOffset = (int) (size / 2);
    switch (type)
    {
        case Math::NORMAL:
        {
            const double N = (double) (size - 1) / 2.0;
            for (Index i = 0, j = -vOffset; i < size; ++i, ++j)
                k(i) = std::exp(-0.5 * ((alpha * j) / N) * ((alpha * j) / N));
            k = k / k.sum();
        }
        break;
        case Math::BOX:
        {
            for (Index i = 0; i < size; ++i)
                k(i) = 1.0 / size;
        }
        break;
        default:
        {
            const double N = ((double) size - 1.0) / 2.0;
            for (Index i = 0, j = -vOffset; i < size; ++i, ++j)
                k(i) = std::exp(-0.5 * ((alpha * j) / N) * ((alpha * j) / N));
            k = k / k.sum();
        }
        break;
    }
    return Math::conv1d(x, k);
}

template<typename Derived>
VectorXd Math::conv1d(const DenseBase<Derived>& x, const VectorXd& k)
{
    if (x.rows() != 1 && x.cols() != 1)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x must be a row or column vector/array." << std::endl;
        throw std::runtime_error(s.str());
    }

    if ((k.size() % 2) == 0)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> Kernel must have an odd number of elements." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (k.sum() < 1.0 - MATH_EPS || k.sum() > 1.0 + MATH_EPS)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << "==> Kernel does not sum 1." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int size = (int) k.size();
    const int offset = (int) (size / 2);
    
    VectorXd y(x.size() + (offset * 2));
#pragma omp parallel for
    for (Index i = 0; i < x.size(); ++i)
        y(i + offset) = x(i);
    for (Index i = 0, j = offset; i < offset; ++i, --j)
        y(i) = x(j);
    for (Index i = y.size() - 1, j = x.size() - offset - 1; i >= y.size() - offset; --i, ++j)
        y(i) = x(j);

    VectorXd z(x.size());
#pragma omp parallel for
    for (int i = 0; i < z.size(); ++i)
        z(i) = y.segment(i, size).transpose() * k;
    
    return z;
}

inline ArrayXd _pchipSlopes(const ArrayXd& h, const ArrayXd& delta)
{
    assert(h.size() == delta.size());
    const int N = h.size();

    ArrayXi k(N);
#pragma omp parallel for
    for (Index i = 0; i < N-1; ++i)
        k(i) = sign(delta(i))*sign(delta(i+1)) > 0 ? i : -1;

    ArrayXd hs(N);
#pragma omp parallel for
    for (Index i = 0; i < N; ++i)
    {
        if (k(i) == -1)
        {
            hs(i) = 0;
            continue;
        }
        hs(i) = h(k(i)) + h(k(i)+1);
    }
    ArrayXd w1(N);
    ArrayXd w2(N);
    ArrayXd dmax(N);
    ArrayXd dmin(N);
#pragma omp parallel for
    for (Index i = 0; i < N; ++i)
    {
        if (k(i) == -1)
        {
            w1(i) = 0.0;
            w2(i) = 0.0;
            dmin(i) = 0.0;
            dmax(i) = 0.0;
            continue;
        }
        w1(i) = (h(k(i)) + hs(i)) / (3.0 * hs(i));
        w2(i) = (h(k(i)+1) + hs(i)) / (3.0 * hs(i));
        dmax(i) = std::abs(delta(k(i)));
        dmin(i) = std::abs(delta(k(i)+1));
        if (dmax(i) < dmin(i))
        {
            dmin(i) = dmax(i);
            dmax(i) = std::abs(delta(k(i)+1));
        }
    }
    ArrayXd slopes(N+1);
#pragma omp parallel for
    for (Index i = 0; i < N; ++i)
    {
        if (k(i) == -1)
        {
            slopes(i) = 0;
            continue;
        }
        slopes(k(i)+1) = dmin(i) / ((w1(i) * delta(k(i)) / dmax(i)) + (w2(i) * delta(k(i)+1) / dmax(i)));
    }
    slopes(0) = ((2.0 * h(0) + h(1)) * delta(0) - h(0) * delta(1)) / (h(0) + h(1));
    if (sign(slopes(0)) != sign(delta(0)))
        slopes(0) = 0;
    else if ((sign(delta(0)) != sign(delta(1))) && (std::abs(slopes(0)) > std::abs(3.0 * delta(0))))
        slopes(0) = 3.0 * delta(0);
    slopes(N) = ((2.0 * h(N-1) + h(N-2)) * delta(N-1) - h(N-1) * delta(N-2)) / (h(N-1) + h(N-2));
    if (sign(slopes(N)) != sign(delta(N-1)))
        slopes(N) = 0;
    else if ((sign(delta(N-1)) != sign(delta(N-2))) && (std::abs(slopes(N)) > std::abs(3.0 * delta(N-1))))
        slopes(N) = 3.0 * delta(N-1);

    return slopes;
}

inline ArrayXd _mAkimaSlopes(const ArrayXd& del)
{
    const int N = del.size();
    
    const double delta_0 = 2.0 * del(0) - del(1);
    const double delta_m1 = 2.0 * delta_0 - del(0);
    const double delta_n = 2.0 * del(N-1) - del(N-2);
    const double delta_n1 = 2.0 * delta_n - del(N-1);

    ArrayXd delta(N+4);
    delta(0) = delta_m1;
    delta(1) = delta_0;
    delta.segment(2, N) = del;
    delta(N+2) = delta_n;
    delta(N+3) = delta_n1;

    const ArrayXd ddelta = Math::diff(delta);
    ArrayXd weights(N+3);
#pragma omp parallel for
    for (Index i = 0; i < N+3; ++i)
        weights(i) = std::abs(ddelta(i)) + std::abs((delta(i) + delta(i+1)) / 2.0);

    ArrayXd slopes(N+1);
#pragma omp parallel for
    for (Index i = 0; i < N+1; ++i)
    {
        const double w = weights(i) + weights(i+2);
        slopes(i) = w == 0 ? 0 : (delta(i+1) * weights(i+2) / w) + (delta(i+2) * weights(i) / w);
    }

    return slopes;
}

inline MatrixXd _pwch(const ArrayXd& x, const ArrayXd& y, const ArrayXd& slopes, const ArrayXd& h, const ArrayXd& delta)
{
    const int N = x.size();

    ArrayXd dzzdx(N-1);
    ArrayXd dzdxdx(N-1);
#pragma omp parallel for
    for (Index i = 0; i < N-1; ++i)
    {
        dzzdx(i) = (delta(i) - slopes(i)) / h(i);
        dzdxdx(i) = (slopes(i+1) - delta(i)) / h(i);
    }
    MatrixXd pp(N-1, 4);
#pragma omp parallel for
    for (Index i = 0; i < N-1; ++i)
    {
        pp(i, 0) = (dzdxdx(i) - dzzdx(i)) / h(i);
        pp(i, 1) = 2.0 * dzzdx(i) - dzdxdx(i);
        pp(i, 2) = slopes(i);
        pp(i, 3) = y(i);
    }
    return pp;
}

inline ArrayXd _ppval(const MatrixXd& pp, const ArrayXd& x, const ArrayXd& xq)
{
    const auto H = Math::histogram(xq, x);
    const VectorXi bin = std::get<2>(H);

    assert(bin.size() == xq.size());

    const int N = xq.size();
    ArrayXd xs(N);
    ArrayXd yq(N);
#pragma omp parallel for
    for (Index i = 0; i < N; ++i)
    {
        xs(i) = xq(i) - x(bin(i));
        yq(i) = pp(bin(i), 0);
    }

    for (Index j = 1; j < 4; ++j)
    {
#pragma omp parallel for
        for (Index i = 0; i < N; ++i)
            yq(i) = xs(i) * yq(i) + pp(bin(i), j);
    }
    return yq;
}


template<typename Derived1, typename Derived2, typename Derived3>
VectorXd Math::interp1(const DenseBase<Derived1>& x, const DenseBase<Derived2>& y, const DenseBase<Derived3> &xq, const Interpolation type)
{
    if (x.size() != y.size())
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x and y must be of the same size." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (x.size() < 3)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> x and y must have more than 2 elements." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (xq.size() < 4)
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> xq must have more than 3 elements." << std::endl;
        throw std::runtime_error(s.str());
    }

    const ArrayXd h = Math::diff(x);
    const ArrayXd delta = Math::diff(y).array() / h;
    const ArrayXd slopes = type == Math::PCHIP ? _pchipSlopes(h, delta) : _mAkimaSlopes(delta);
    
    return _ppval(_pwch(x, y, slopes, h, delta), x, xq);
}

/***************************** Implementation of Histogram Edges class *****************************/
/***************************** must be here with the declaration first *****************************/

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
    for (Index i = 0; i < nbinsActual; ++i)
        y(i) = leftEdge + ((double) i * binWidth);
    y(nbinsActual) = rightEdge;
    return y;
}

template<typename Derived>
VectorXd Edges::autorule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    const typename Derived::Scalar xrange = maxx - minx;
    if (x.size() > 0 && std::is_integral<typename Derived::Scalar>::value && xrange <= 50 && maxx <= (MATH_FLINTMAX / 2.0) && minx >= -(MATH_FLINTMAX / 2.0))
        return Edges::integerrule(x, minx, maxx, hardlimits);
    else
        return Edges::scottsrule(x, minx, maxx, hardlimits);
}

template<typename Derived>
VectorXd Edges::scottsrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    const double sigma = Math::std(x);
    const double binwidth = 3.5 * sigma / std::pow((double) x.size(), 1.0 / 3.0);
    if (hardlimits)
    {
        const int nbins = (int) std::ceil((double) (maxx - minx) / binwidth);
        return Math::linspace(minx, maxx, nbins);
    }
    else
    {
        return Edges::binpicker(minx, maxx, binwidth);
    }
}

template<typename Derived>
VectorXd Edges::integerrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    if (maxx > (MATH_FLINTMAX / 2.0) || minx < -(MATH_FLINTMAX / 2.0))
    {
        std::stringstream s;
        s << __PRETTY_FUNCTION__ << " ==> Input out of integer range." << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const int maximumBins = 65536;
    const typename Derived::Scalar xscale = x.derived().array().abs().maxCoeff();
    const typename Derived::Scalar xrange = x.maxCoeff() - x.minCoeff();
    
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

template<typename Derived>
VectorXd Edges::sturgesrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    const int nbins = std::max(std::ceil(std::log2(x.size()) + 1), 1);
    if (hardlimits)
    {
        return Math::linspace(minx, maxx, nbins);
    }
    else
    {
        return Edges::binpicker(minx, maxx, (double)(maxx - minx) / (double) nbins);
    }
}

template<typename Derived>
VectorXd Edges::sqrtrule(const DenseBase<Derived>& x, const typename Derived::Scalar minx, const typename Derived::Scalar maxx, const bool hardlimits)
{
    const int nbins = std::max(std::ceil(std::sqrt(x.size())), 1);
    if (hardlimits)
    {
        return Math::linspace(minx, maxx, nbins);
    }
    else
    {
        return Edges::binpicker(minx, maxx, (double)(maxx - minx) / (double) nbins);
    }
}

#endif