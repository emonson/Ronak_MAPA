/*
 * cs_ioutils.h
 *
 *  Created on: Apr 19, 2009
 *      Author: user
 */

#ifndef CS_IOUTILS_H_
#define CS_IOUTILS_H_


#include <vector>
#include <algorithm>
#include <utility>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>


#endif /* CS_IOUTILS_H_ */

// Load a matrix from a text file
int cs_LoadBinaryNetworkFromFile(FILE *cFile, cs **vA, unsigned long int M=0, unsigned long int N=0, int nnz=0, int HeaderLines=0, bool ToSymm=false);
int cs_LoadFromFile(FILE *cFile, cs **vA, unsigned long int M=0, unsigned long int N=0, int nnz=0, int HeaderLines=0, bool ToSymm=false);
int cs_LoadFromFile(const char *cFileName, cs **vA, unsigned long int M=0, unsigned long int N=0, int nnz=0, int HeaderLines=0, bool ToSymm=false);
// Computes the L^2 norm of the columns of a matrix
double *cs_col2norms (const cs *A);
// Rank-revealing sparse QR
int cs_gsqr (const cs *A);
// Find column of maximum L^2 norm
double cs_max2norm (const cs *A);
// Sums of columns of sparse matrix
double *cs_colsums(const cs *A);
// Find maximum entry of a vector
int cs_vecfindmax (const double *v, const int vlen, int *maxidx);
// Inserts column in a matrix
int cs_insertlastcolumn(cs *T, const CS_INT *i, const CS_INT j, const CS_ENTRY *x, const CS_INT n);
// Computes inner products of columns of a matrix with vector
int cs_scolip( const cs*A, const cs*v, double** sum);
// Extract column from matrix
int cs_getcol(const cs*A, int colidx, cs **col);
// Copy sparse matrix
int cs_copy(const cs*A, cs **B);
// Subtract columns from matrix
int cs_subcols(const cs*A, cs**Col, double*b);
// Applies exponential function to entries of A
int cs_applyexp(cs *A, double sigma);
// Maps a symmetric matrix A to (1/sqrt(D))*A*(1/sqrt(D))
int cs_symminvsqrtdeg(cs *A);
// Maps a symmetric matrix A to (1/sqrt(D))*(D+A)*(1/sqrt(D))
cs *cs_W2DinvsqrtDpWDinvsqrt(cs *A);
// Construct n by n identity matrix
cs* cs_Id(CS_INT n);
// Construct diagonal matrix with given entries
cs* cs_Diag(CS_INT n, double* diag);
// Remove entries in the matrix that are repeated twice, replacing with the average
int cs_ave_dupl (cs *A);


using namespace boost;

typedef adjacency_list <vecS, vecS, undirectedS> uGraph;

// Converts a cs sparse matrix in a Graph object
uGraph* ConvertCSparseToGraph(cs *A);
