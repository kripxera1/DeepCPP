/* 
 * File: src/algebra.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains definitions for basic algebraic operations.
 */

#include "algebra.h"
#include <iostream>


Matrix dot(const Matrix& a, const Matrix& b) {

    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();

    Matrix c(n, Vector(p, 0));

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m; ++k) {
            for (int j = 0; j < p; ++j) {
                #pragma omp atomic update
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return c;
}


//sums vector b with each column of the matrix M
//Pre: M.size() == b.size();
Matrix sum(const Matrix & M, const Vector & b) {
    int num_rows = M.size();
    int num_cols = M[0].size();

    Matrix M2(num_rows, Vector(num_cols, 0));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            M2[i][j] = b[i] + M[i][j];
        }
    }

    return M2;
}


Matrix T(const Matrix& m) {
    int num_rows = m.size();
    int num_cols = m[0].size();

    Matrix mt(num_cols, Vector(num_rows, 0));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            mt[j][i] = m[i][j];
        }
    }

    return mt;
}


Matrix minusM(const Matrix &m1, const Matrix &m2) {
    int num_rows = m1.size();
    int num_cols = m1[0].size();

    Matrix m(num_rows, Vector(num_cols, 0));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            m[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return m;
}


Matrix product(Matrix m1, double a) {
    int num_rows = m1.size();
    int num_cols = m1[0].size();

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            m1[i][j] *= a;
        }
    }

    return m1;
}


Vector product(Vector v, double a) {
    int vecSize = v.size();

    #pragma omp parallel for
    for (int i = 0; i < vecSize; i++) {
        v[i] *= a;
    }

    return v;
}


//devuelve un vector con la suma de cada una de las filas
Vector rowsSum(const Matrix & m) {
    int num_rows = m.size();
    int num_cols = m[0].size();

    Vector v(num_rows, 0);

    #pragma omp parallel for
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            #pragma omp atomic
            v[i] += m[i][j];
        }
    }

    return v;
}


Matrix hadamard(const Matrix &m1, const Matrix &m2) {
    int num_rows = m1.size();
    int num_cols = m1[0].size();

    Matrix m(num_rows, Vector(num_cols, 0));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            m[i][j] = m1[i][j] * m2[i][j];
        }
    }

    return m;
}
