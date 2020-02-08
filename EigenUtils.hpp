/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                          */
/* Universidad Politecnica de Valencia, Spain                              */
/*                                                                         */
/* Copyright (C) 2014 Javier Juan Albarracin                               */
/*                                                                         */
/***************************************************************************
* Eigen Utils                                                              *
***************************************************************************/

#ifndef EIGENUTILS_HPP
#define EIGENUTILS_HPP

#include <Eigen/Dense>

using namespace Eigen;

class EigenUtils
{
public:
	template <typename Derived>
	inline static void removeRow(DenseBase<Derived> &matrix, size_t rowToRemove);
	template <typename Derived>
	inline static void removeColumn(DenseBase<Derived> &matrix, size_t colToRemove);
};

template <typename Derived>
void EigenUtils::removeRow(DenseBase<Derived> &matrix, size_t rowToRemove)
{
    size_t numRows = matrix.rows() - 1;
    size_t numCols = matrix.cols();

    if (rowToRemove < numRows)
        matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

    matrix.derived().conservativeResize(numRows, numCols);
}

template <typename Derived>
void EigenUtils::removeColumn(DenseBase<Derived> &matrix, size_t colToRemove)
{
    size_t numRows = matrix.rows();
    size_t numCols = matrix.cols() - 1;

    if (colToRemove < numCols)
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

    matrix.derived().conservativeResize(numRows, numCols);
}

#endif