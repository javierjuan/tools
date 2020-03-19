/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* SVFMM <-> Python data type conversions                                   *
***************************************************************************/

#ifndef PYSVFMM_HPP
#define PYSVFMM_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <SpatiallyVariantFiniteMixtureModel.hpp>
#include <Distributions/MultivariateNormal.hpp>
#include <Distributions/MultivariateTStudent.hpp>
#include <Distributions/Gamma.hpp>
#include <MarkovRandomFields/MarkovRandomField.hpp>
#include <MarkovRandomFields/FrequentistGaussMarkovRandomField.hpp>
#include <MarkovRandomFields/FrequentistTStudentMarkovRandomField.hpp>
#include <MarkovRandomFields/FrequentistNonLocalMarkovRandomField.hpp>
#include <MarkovRandomFields/BayesianGaussMarkovRandomField.hpp>
#include <MarkovRandomFields/BayesianTStudentMarkovRandomField.hpp>
#include <MarkovRandomFields/BayesianNonLocalMarkovRandomField.hpp>
#include <PyCImg.hpp>
#include <PySTL.hpp>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <string>
#include <omp.h>

namespace py = pybind11;
using namespace cimg_library;
using namespace Eigen;

template<typename T>
using ndarray = py::array_t<T, py::array::c_style | py::array::forcecast>;


template<typename Distribution, int Dimensions>
ndarray<double> muToNumpy(const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> *svfmm)
{
    const int K = svfmm->components();
    const int D = svfmm->spectrum();

    ndarray<double> output({K, D});
    py::buffer_info info = output.request();

    double *ptr = (double*) info.ptr;
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            ptr[j * K + i] = (*svfmm)[i].mu()(j);
        }
    }

    return output;
}


template<typename Distribution, int Dimensions>
ndarray<double> sigmaToNumpy(const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> *svfmm)
{
    const int K = svfmm->components();
    const int D = svfmm->spectrum();

    ndarray<double> output({D, D, K});
    py::buffer_info info = output.request();

    double *ptr = (double*) info.ptr;
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            for (int k = 0; k < D; ++k)
            {
                ptr[(i * D * D) + (k * D) + j] = (*svfmm)[i].sigma()(j, k);
            }
        }
    }
    return output;
}


template<int Dimensions>
ndarray<double> nuToNumpy(const SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> *svfmm)
{
    const int K = svfmm->components();

    ndarray<double> output({K});
    py::buffer_info info = output.request();
    
    double *ptr = (double*) info.ptr;
    for (int i = 0; i < K; ++i)
    {
        ptr[i] = (*svfmm)[i].nu();
    }
    
    return output;
}


template<int Dimensions>
ndarray<double> kToNumpy(const SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> *svfmm)
{
    const int K = svfmm->components();

    ndarray<double> output({K});
    py::buffer_info info = output.request();
    
    double *ptr = (double*) info.ptr;
    for (int i = 0; i < K; ++i)
    {
        ptr[i] = (*svfmm)[i].k();
    }
    
    return output;
}


template<int Dimensions>
ndarray<double> thetaToNumpy(const SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> *svfmm)
{
    const int K = svfmm->components();

    ndarray<double> output({K});
    py::buffer_info info = output.request();
    
    double *ptr = (double*) info.ptr;
    for (int i = 0; i < K; ++i)
    {
        ptr[i] = (*svfmm)[i].theta();
    }
    
    return output;
}

/***************************** Python SVFMM Base *****************************/

class PySVFMMBase
{
public:
    enum Mode { SEGMENTATION, MIXTURE, COMPLETE };
};


/***************************** Python SVFMM *****************************/

template<typename Distribution, int Dimensions>
class PySVFMM : public PySVFMMBase
{
public:
    static py::dict toPython(const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> *svfmm, const CImg<bool> &mask, const PySVFMMBase::Mode mode);
};

template<typename Distribution, int Dimensions>
py::dict PySVFMM<Distribution, Dimensions>::toPython(const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> *svfmm, const CImg<bool> &mask, const PySVFMMBase::Mode mode)
{
    std::stringstream s;
    s << "In function " << __PRETTY_FUNCTION__ << " ==> Function must be specialized for each distribution type." << std::endl;
    throw std::runtime_error(s.str());

    return py::dict();
}

/***************************** Template Specialization Multivariate Normal *****************************/

template<int Dimensions>
class PySVFMM<MultivariateNormal, Dimensions> : public PySVFMMBase
{
public:
    static py::dict toPython(const SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> *svfmm, const CImg<bool> &mask, const PySVFMMBase::Mode mode);
};

template<int Dimensions>
py::dict PySVFMM<MultivariateNormal, Dimensions>::toPython(const SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> *svfmm, const CImg<bool> &mask, const PySVFMMBase::Mode mode)
{
    py::dict output = py::dict();

    if (mode == SEGMENTATION || mode == MIXTURE || mode == COMPLETE)
    {
        output["segmentation"] = PyCImg::toNumpy(EigenCImg::toCImg(svfmm->labels().array() + 1, mask));
    }
    
    if (mode == MIXTURE || mode == COMPLETE)
    {
        output["mu"] = muToNumpy(svfmm);
        output["sigma"] = sigmaToNumpy(svfmm);
        output["priors"] = PyCImg::toNumpy(EigenCImg::toCImg(svfmm->coefficients(), mask));
        output["posteriors"] = PyCImg::toNumpy(EigenCImg::toCImg(svfmm->posteriorProbabilities(), mask));
        output["loglikelihood"] = PySTL::toNumpy(svfmm->logLikelihoodHistory());

    }

    if (mode == COMPLETE)
    {
        py::dict mrf = py::dict();

        const MarkovRandomFieldBase::Model model = svfmm->spatialCoefficients().get()->model();
        const MarkovRandomFieldBase::Tropism tropism = svfmm->spatialCoefficients().get()->tropism();
        const MarkovRandomFieldBase::Topology topology = svfmm->spatialCoefficients().get()->topology();
        const MarkovRandomFieldBase::Estimation estimation = svfmm->spatialCoefficients().get()->estimation();

        mrf["cliques"] = svfmm->spatialCoefficients().get()->cliques();
        mrf["connectivity"] = svfmm->spatialCoefficients().get()->connectivity();
        mrf["nodes"] = svfmm->spatialCoefficients().get()->nodes();
        mrf["classes"] = svfmm->spatialCoefficients().get()->classes();
        mrf["model"] = model == MarkovRandomFieldBase::GAUSS ? "gaussian" : (MarkovRandomFieldBase::TSTUDENT ? "tstudent" : "nonlocal");
        mrf["tropism"] = tropism == MarkovRandomFieldBase::ISOTROPIC ? "isotropic" : "anisotropic";
        mrf["topology"] = topology == MarkovRandomFieldBase::ORTHOGONAL ? "orthogonal" : "complete";
        mrf["estimation"] = estimation == MarkovRandomFieldBase::FREQUENTIST ? "frequentist" : "bayesian";
        
        const MarkovRandomField<Dimensions> *ptr_ = svfmm->spatialCoefficients().get();

        // Gauss
        if (model == MarkovRandomFieldBase::GAUSS && tropism == MarkovRandomFieldBase::ISOTROPIC && estimation == MarkovRandomFieldBase::BAYESIAN)
        {
            const BayesianIsotropicGaussMarkovRandomField<Dimensions> *ptr = static_cast<const BayesianIsotropicGaussMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
        }
        if (model == MarkovRandomFieldBase::GAUSS && tropism == MarkovRandomFieldBase::ISOTROPIC && estimation == MarkovRandomFieldBase::FREQUENTIST)
        {
            const FrequentistIsotropicGaussMarkovRandomField<Dimensions> *ptr = static_cast<const FrequentistIsotropicGaussMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
        }
        if (model == MarkovRandomFieldBase::GAUSS && tropism == MarkovRandomFieldBase::ANISOTROPIC && estimation == MarkovRandomFieldBase::BAYESIAN)
        {
            const BayesianAnisotropicGaussMarkovRandomField<Dimensions> *ptr = static_cast<const BayesianAnisotropicGaussMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
        }
        if (model == MarkovRandomFieldBase::GAUSS && tropism == MarkovRandomFieldBase::ANISOTROPIC && estimation == MarkovRandomFieldBase::FREQUENTIST)
        {
            const FrequentistAnisotropicGaussMarkovRandomField<Dimensions> *ptr = static_cast<const FrequentistAnisotropicGaussMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
        }
        // T-student
        if (model == MarkovRandomFieldBase::TSTUDENT && tropism == MarkovRandomFieldBase::ISOTROPIC && estimation == MarkovRandomFieldBase::BAYESIAN)
        {
            const BayesianIsotropicTStudentMarkovRandomField<Dimensions> *ptr = static_cast<const BayesianIsotropicTStudentMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
        }
        if (model == MarkovRandomFieldBase::TSTUDENT && tropism == MarkovRandomFieldBase::ISOTROPIC && estimation == MarkovRandomFieldBase::FREQUENTIST)
        {
            const FrequentistIsotropicTStudentMarkovRandomField<Dimensions> *ptr = static_cast<const FrequentistIsotropicTStudentMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
        }
        if (model == MarkovRandomFieldBase::TSTUDENT && tropism == MarkovRandomFieldBase::ANISOTROPIC && estimation == MarkovRandomFieldBase::BAYESIAN)
        {
            const BayesianAnisotropicTStudentMarkovRandomField<Dimensions> *ptr = static_cast<const BayesianAnisotropicTStudentMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
        }
        if (model == MarkovRandomFieldBase::TSTUDENT && tropism == MarkovRandomFieldBase::ANISOTROPIC && estimation == MarkovRandomFieldBase::FREQUENTIST)
        {
            const FrequentistAnisotropicTStudentMarkovRandomField<Dimensions> *ptr = static_cast<const FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
        }
        // Non-Local
        if (model == MarkovRandomFieldBase::NONLOCAL && tropism == MarkovRandomFieldBase::ISOTROPIC && estimation == MarkovRandomFieldBase::BAYESIAN)
        {
            const BayesianIsotropicNonLocalMarkovRandomField<Dimensions> *ptr = static_cast<const BayesianIsotropicNonLocalMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
            mrf["patch_size"] = ptr->patchSize();
            mrf["chi2_sigma"] = ptr->chi2Sigma();
        }
        if (model == MarkovRandomFieldBase::NONLOCAL && tropism == MarkovRandomFieldBase::ISOTROPIC && estimation == MarkovRandomFieldBase::FREQUENTIST)
        {
            const FrequentistIsotropicNonLocalMarkovRandomField<Dimensions> *ptr = static_cast<const FrequentistIsotropicNonLocalMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
            mrf["patch_size"] = ptr->patchSize();
            mrf["chi2_sigma"] = ptr->chi2Sigma();
        }
        if (model == MarkovRandomFieldBase::NONLOCAL && tropism == MarkovRandomFieldBase::ANISOTROPIC && estimation == MarkovRandomFieldBase::BAYESIAN)
        {
            const BayesianAnisotropicNonLocalMarkovRandomField<Dimensions> *ptr = static_cast<const BayesianAnisotropicNonLocalMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
            mrf["patch_size"] = ptr->patchSize();
            mrf["chi2_sigma"] = ptr->chi2Sigma();
        }
        if (model == MarkovRandomFieldBase::NONLOCAL && tropism == MarkovRandomFieldBase::ANISOTROPIC && estimation == MarkovRandomFieldBase::FREQUENTIST)
        {
            const FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions> *ptr = static_cast<const FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions>*>(ptr_);
            mrf["sigma"] = ptr->sigma();
            mrf["nu"] = ptr->nu();
            mrf["patch_size"] = ptr->patchSize();
            mrf["chi2_sigma"] = ptr->chi2Sigma();
        }

        output["mrf"] = mrf;
    }

    return output;
}

#endif