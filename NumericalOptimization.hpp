/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                          */
/* Universidad Politecnica de Valencia, Spain                              */
/*                                                                         */
/* Copyright (C) 2014 Javier Juan Albarracin                               */
/*                                                                         */
/***************************************************************************
* Numerical Optimization                                                   *
***************************************************************************/

#ifndef NUMERICALOPTIMIZATION_HPP
#define NUMERICALOPTIMIZATION_HPP

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cmath>

#define __STR_FUNCNAME__  std::string(__FUNCTION__)

class NumericalOptimization
{
public:
	static double NewtonRaphsonSolver(double(*f) (const double, const double), double(*fprime) (const double), const double K, double x0);
	static double BisectionRootSolver(double(*f) (const double, const double), const double K, double a, double b);
	static double GoldenSectionSearch(double(*f) (const double, const double), const double K, double a, double b);
};

/***************************** Implementation *****************************/

double NumericalOptimization::NewtonRaphsonSolver(double(*f)(const double, const double), double(*fprime) (const double), const double K, double x0)
{
	const double tolerance = 1e-7;
	const double epsilon = 1e-14;
	const int maxIterations = 500;
	
	double x1 = x0;
	
	for (int i = 0; i < maxIterations; ++i)
	{
		const double y = f(x0, K);
		const double yprime = fprime(x0);

		if (abs(yprime) < epsilon)
		{
			std::stringstream s;
			s << __STR_FUNCNAME__ << " ==> Tolerance reached without convergence" << std::endl;
			throw std::runtime_error(s.str());
		}

		x1 = x0 - y / yprime;

		if (abs(x1 - x0) / abs(x1) < tolerance)
			return x1;
		x0 = x1;
	}
	std::stringstream s;
	s << __STR_FUNCNAME__ << " ==> Maximum number of iterations reached without convergence" << std::endl;
	throw std::runtime_error(s.str());
}

double NumericalOptimization::BisectionRootSolver(double(*f) (const double, const double), const double K, double a, double b)
{
	const double tolerance = 1e-7;
	const int maxIterations = 500;
	
	for (int i = 0; i < maxIterations; ++i)
	{
		const double c = (a + b) / 2.0;
		const double fc = f(c, K);
		const double fa = f(a, K);

		if (fc == 0 || ((b - a) / 2.0) < tolerance)
			return c;

		if ((fc >= 0) ^ (fa < 0))
			a = c;
		else
			b = c;
	}
	std::stringstream s;
	s << __STR_FUNCNAME__ << " ==> Maximum number of iterations reached without convergence" << std::endl;
	throw std::runtime_error(s.str());
}

double NumericalOptimization::GoldenSectionSearch(double(*f)(const double, const double), const double K, double a, double b)
{
	const double tolerance = 1e-7;
	const double goldenRatio = 1.618033988749895;
	
	double c = b - (b - a) / goldenRatio;
	double d = a + (b - a) / goldenRatio;
    
	while (fabs(c - d) > tolerance)
	{
        if (f(c, K) < f(d, K))
			b = d;
        else
			a = c;

		c = b - (b - a) / goldenRatio;
		d = a + (b - a) / goldenRatio;
	}
    return (b + a) / 2.0;
}

#endif