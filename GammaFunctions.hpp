/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2018 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Gamma Functions                                                          *
***************************************************************************/

#ifndef GAMMAFUNCTIONS_HPP
#define GAMMAFUNCTIONS_HPP

#include <stdexcept>
#include <cmath>

#define GAMMAFUNCTIONS_PI 3.141592653589793
#define GAMMAFUNCTIONS_EULER 0.577215664901532
#define GAMMAFUNCTIONS_LOG2 0.693147180559945
#define GAMMAFUNCTIONS_ZERO 1e-40
#define GAMMAFUNCTIONS_INF 1e+150

class GammaFunctions
{
public:
	// Gamma functions
	inline static double gamma(const int x);
	inline static double gamma(const double x);
	
	// Log gamma functions
	inline static double logGamma(const double x);
	
	// Digamma functions
	inline static double digamma(const int x);
	inline static double digamma(const double x);
	
	// Trigamma functions
	inline static double trigamma(const double x);
};

static const double dg_coeff[10] =
{
-.83333333333333333e-1, .83333333333333333e-2,-.39682539682539683e-2, 
 .41666666666666667e-2,-.75757575757575758e-2, .21092796092796093e-1, 
-.83333333333333333e-1, .4432598039215686, -.3053954330270122e+1, .125318899521531e+2
};
									 
static const double lgg_coeff[11] = 
{ 
1.000000000000000174663, 5716.400188274341379136, -14815.30426768413909044,
14291.49277657478554025, -6348.160217641458813289, 1301.608286058321874105,
-108.1767053514369634679, 2.605696505611755827729, -0.7423452510201416151527e-2,
0.5384136432509564062961e-7, -0.4023533141268236372067e-8
};

double GammaFunctions::gamma(const int x)
{
	if (x <= 0 || x > 100)
		return GAMMAFUNCTIONS_INF;
	
	double g = 1.0;
	for (int i = 2; i < x; ++i)
		g *= i;
	return g;
}

double GammaFunctions::gamma(const double x)
{
	if (abs(x - (int) x) <= GAMMAFUNCTIONS_ZERO)
		return GammaFunctions::gamma((int) x);
	else if (x > 0)
		return exp(GammaFunctions::logGamma(x));
	else
	{
		double xp = x;
		double xf = 1.0;
		
		while (xp < -1.0)
			xf /= xp++;
		return xf * GAMMAFUNCTIONS_PI / (exp(GammaFunctions::logGamma(1.0 - xp)) * sin(GAMMAFUNCTIONS_PI * xp));
	}
}

double GammaFunctions::logGamma(const double x)
{
	if (x <= 0 || x > 100)
		return GAMMAFUNCTIONS_INF;
	
	double xm = x + 9;
	double ss = xm - 0.5;
	double ser = lgg_coeff[0];
	
	for (int i = 10; i > 0; --i, xm -= 1.0)
		ser += lgg_coeff[i] / xm;
	
	return (x - 0.5) * log(ss) - ss + log(ser * sqrt(2.0 * GAMMAFUNCTIONS_PI));
}

double GammaFunctions::digamma(const int x)
{
	if (x < 0)
		throw std::runtime_error("Digamma function not defined for negative values");
	if (x == 0)
		return -GAMMAFUNCTIONS_INF;
	
	double s = -GAMMAFUNCTIONS_EULER;
	for (int i = 1; i < x; ++i)
		s += 1.0 / (double) i;
	
	return s;
}

double GammaFunctions::digamma(const double x)
{
	double xa = fabs(x);
	double dgam = 0.0;
	
	if (fabs(xa - (int) xa) <= GAMMAFUNCTIONS_ZERO)
		return digamma((int) x);
	else if (fabs((xa + 0.5) - (int)(xa + 0.5)) <= GAMMAFUNCTIONS_ZERO)
	{
		const int n = (int) (xa - 0.5);
		for (int i = 1; i <= n; ++i)
			dgam +=  1.0 / (i + i - 1.0);
		dgam = 2.0 * (dgam - GAMMAFUNCTIONS_LOG2) - GAMMAFUNCTIONS_EULER;
	}
	else
	{
		if (xa < 10)
		{
			const int n = 10 - (int) xa;
			for (int i = 0; i < n; ++i)
				dgam -= 1.0 / (i + xa);
			xa += n;
		}
		const double overx2 = 1.0 / (xa * xa);
		double overx2k = overx2;
		dgam += log(xa) - 0.5 / xa;
		for (int i = 0; i < 10; ++i, overx2k *= overx2)
			dgam += dg_coeff[i] * overx2k;
	}
	if (x < 0)
		dgam -= GAMMAFUNCTIONS_PI * (tan(GAMMAFUNCTIONS_PI * x)) + 1.0 / x;
	return dgam;
}

double GammaFunctions::trigamma(const double x)
{
	if (x < 0)
		throw std::runtime_error("Trigamma function not defined for negative values");
	if (x == 0)
		return GAMMAFUNCTIONS_INF;
	if (x <= 0.0001)
		return (1.0 / x / x);
	
	double z = x;
	double value = 0.0;
	while (z < 5)
	{
		value += 1.0 / z / z;
		z += 1.0;
	}

	const double y = 1.0 / z / z;
	value += 0.5 * y + ( 1.0 + y * (0.1666666667 + y * (-0.03333333333 + y * (0.02380952381 + y * (-0.03333333333))))) / z;
	
	return value;
}

#endif