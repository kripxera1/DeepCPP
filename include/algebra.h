/* 
 * File: include/algebra.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains declarations for basic algebraic operations.
 */

#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <vector>
#include "typedefs.h"


Vector rowsSum(const Matrix & M);

Vector product(Vector v, double a);

Matrix product(Matrix M, double a);

Matrix T(const Matrix&M);

Matrix dot(const Matrix & M1, const Matrix & M2);

Matrix sum(const Matrix & M, const Vector & b);

Matrix minusM(const Matrix & M1, const Matrix & M2);

Matrix hadamard(const Matrix & M1, const Matrix & M2);


#endif