//
//  SVD.h
//  GeometricMultiResolutionAnalysis
//
//  Created by Mauro Maggioni on 8/9/12.
//  Copyright (c) 2012 Mauro Maggioni. All rights reserved.
//

#ifndef __GeometricMultiResolutionAnalysis__SVD__
#define __GeometricMultiResolutionAnalysis__SVD__

#include <iostream>
#include <pthread.h>
#include <Accelerate/Accelerate.h>          // For BLAS and LAPACK
#include "ANN_utils.h"
#include "TimeUtils.h"

// Some macros
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))



// Computes multiscale SVD - single threaded
typedef struct {
	unsigned long int nPts; // Number of points
	unsigned long int nDims; // Dimensionality of ambient space
	unsigned long int NumberOfScales; // Number of scales
	bool RadiusMode; // Whether to use multiscale radii or nearest neighbors
	bool spaceAllocated; // Whether memory for return values is already allocated or not
	unsigned long int startIdx; // Only look at points from startIdx to endIdx
	unsigned long int endIdx;
	//unsigned short int nThreads;	// Number of threads for the multiscale svd computation
    // Ronak added //
 //   double             *S;
} param_MSVD;


typedef struct {
    ANNpointArray       dataPts;
    ANNidxArray        *nnIdxs;
    unsigned long int  *ChosenPtIdxs;
    unsigned long int   nDims;
    unsigned long int   nPts;
    unsigned long int   nnIdxsLen;
    unsigned long int  *nnIdxsWidth;
    unsigned long       maxIdxsWidth;
    unsigned int        nThreads;
    double             *S;
    unsigned long int   id;
    TimeList           *timings;
    
} param_MSVD_single;


int Cov_SVD(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *V);
int Cov_SVD_withV(double* Pts, unsigned long nPts, unsigned long nDim,double *V);
int Cov_SVD_withU(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *U);

//int Cov_SVD_withU(vector<vector<float>>& sPts, unsigned long nPts, unsigned long nDim, double *S, double *U);

void *ComputeMSVD_multi(void *param);
int MultiSVDOnSetOfPoints( param_MSVD_single *params );
void *MultiSVDOnSetOfPoints_multi(void *param);
unsigned long int max(double* array, unsigned long int len);




#endif /* defined(__GeometricMultiResolutionAnalysis__SVD__) */
