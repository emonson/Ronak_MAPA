//
//  SVD.cpp
//  GeometricMultiResolutionAnalysis
//
//  Created by Mauro Maggioni on 8/9/12.
//  Copyright (c) 2012 Mauro Maggioni. All rights reserved.
//
#include <iostream.h>
#include <mach/mach_time.h>
#include "SVD.h"

#define MULTISCALESVDONSETOFPOINTS_AVOIDMEMORYCLASHES       0

//
// Computes svd of covariance matrix of points
//
//	IN:
//   Pts    : a nPts by nDim matrix of points
//
//	OUT:
//   return : info as returned by dgesvd_ (in particular, 0 is success)
//   S      : pointer to at least min(nPts,nDim) doubles to contain the s.v.
//   V      : pointer to min(nDim,nPts)*nDim matrix of rows of V^T ,  ronak added:A: all N rows of V^T are returned in array VT.
// 
int Cov_SVD(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *V) {
    __CLPK_integer info = 0;
//double *V;
	if( MIN(nDim,nPts)==0 )
        return info;
    
double*   MeanPt = new double[nDim];
    
    // Compute the mean of the points
    unsigned long int k, j;
    for (j = 0; j < (unsigned long) nDim; j++) {
        MeanPt[j] = 0;
        for (k = 0; k < (unsigned long) nPts; k++)
            MeanPt[j] += Pts[k * nDim + j];
        MeanPt[j] /= nPts;
    }
    
    // Center the points and normalize by 1/\sqrt{n}
    double sqrtnPts = sqrt((double) nPts);
    for (k = 0; k < (unsigned long) nPts; k++)
        for (j = 0; j < (unsigned long) nDim; j++)
            Pts[k * nDim + j] = (Pts[k * nDim + j] - MeanPt[j]) / sqrtnPts;
    
    
    // Calculate SVD
    __CLPK_integer m = (__CLPK_integer)nDim; // rows
    __CLPK_integer n = (__CLPK_integer)nPts; // coloumns
    
    __CLPK_integer lapack_workl = 20*MIN(m,n);
    uint64_t clock_start = mach_absolute_time();
    __CLPK_doublereal *lapack_work = (__CLPK_doublereal*)malloc(lapack_workl*sizeof(__CLPK_doublereal));
  //  double MemoryAllocation_t = subtractTimes( mach_absolute_time(), clock_start );
 //   char lapack_param[1] = {'S'}; //the first min(m,n) rows of V**T (the right singular vectors) are returned in the array VT;

    char lapack_param[1] = {'A'}; //the first min(m,n) rows of V**T (the right singular vectors) are returned in the array VT;
    char lapack_param1[1] = {'n'};

    clock_start = mach_absolute_time();
    dgesvd_(lapack_param1, lapack_param, &m, &n, Pts, &m, S, NULL, &m, V, &n, lapack_work, &lapack_workl, &info);
 //   double dgesvd_t = subtractTimes( mach_absolute_time(), clock_start );
        
    // Handle error conditions
    if (info)
        printf("Could not compute SVD with error %d\n", info);
    else    {
        /*        printf("\n Solution is:\n");
         for( unsigned int k = 0; k<NUM_VARIABLES; k++ )
         printf("%f,", S[k]);
         printf("\n");*/
    }
    
    free( lapack_work );
    
	return info;
}

int Cov_SVD_withV(double* Pts, unsigned long nPts, unsigned long nDim,double *S, double *V) {
   
   // double *S;
    __CLPK_integer info = 0;
    
	if( MIN(nDim,nPts)==0 )
        return info;
    
    double*   MeanPt = new double[nDim];
    
    // Compute the mean of the points
    unsigned long int k, j;
    for (j = 0; j < (unsigned long) nDim; j++) {
        MeanPt[j] = 0;
        for (k = 0; k < (unsigned long) nPts; k++)
            MeanPt[j] += Pts[k * nDim + j];
        MeanPt[j] /= nPts;
    }
    
    // Center the points and normalize by 1/\sqrt{n}
    double sqrtnPts = sqrt((double) nPts);
    for (k = 0; k < (unsigned long) nPts; k++)
        for (j = 0; j < (unsigned long) nDim; j++)
            Pts[k * nDim + j] = (Pts[k * nDim + j] - MeanPt[j]) / sqrtnPts;
    
    
    // Calculate SVD
    __CLPK_integer m = (__CLPK_integer)nDim; // rows
    __CLPK_integer n = (__CLPK_integer)nPts; // coloumns
    
    __CLPK_integer lapack_workl = 20*MIN(m,n);
    uint64_t clock_start = mach_absolute_time();
    __CLPK_doublereal *lapack_work = (__CLPK_doublereal*)malloc(lapack_workl*sizeof(__CLPK_doublereal));
//    double MemoryAllocation_t = subtractTimes( mach_absolute_time(), clock_start );
    char lapack_param[1] = {'A'}; //the first min(m,n) rows of V**T (the right singular vectors) are returned in the array VT;
    char lapack_param1[1] = {'n'};
    
    clock_start = mach_absolute_time();
    dgesvd_(lapack_param1, lapack_param, &m, &n, Pts, &m, S, NULL, &m, V, &n, lapack_work, &lapack_workl, &info);
 //   double dgesvd_t = subtractTimes( mach_absolute_time(), clock_start );
    
    // Handle error conditions
    if (info)
        printf("Could not compute SVD with error %d\n", info);
    else    {
        /*        printf("\n Solution is:\n");
         for( unsigned int k = 0; k<NUM_VARIABLES; k++ )
         printf("%f,", S[k]);
         printf("\n");*/
    }
    
    free( lapack_work );
    
	return info;
}



int Cov_SVD_withU(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *U) {
    
    __CLPK_integer info = 0;
    
	if( MIN(nDim,nPts)==0 )
        return info;
    // whether centering or not make difference......
  /*  double*   MeanPt = new double[nDim];
    
    // Compute the mean of the points
    unsigned long int k, j;
    for (j = 0; j < (unsigned long) nDim; j++) {
        MeanPt[j] = 0;
        for (k = 0; k < (unsigned long) nPts; k++)
            MeanPt[j] += Pts[k * nDim + j];
        MeanPt[j] /= nPts;
    }
    
    // Center the points and normalize by 1/\sqrt{n}
    double sqrtnPts = sqrt((double) nPts);
    for (k = 0; k < (unsigned long) nPts; k++)
        for (j = 0; j < (unsigned long) nDim; j++)
            Pts[k * nDim + j] = (Pts[k * nDim + j] - MeanPt[j]) / sqrtnPts;*/
    
    
    // Calculate SVD
    __CLPK_integer m = (__CLPK_integer)nDim; // rows
    __CLPK_integer n = (__CLPK_integer)nPts; // coloumns
    
    __CLPK_integer lapack_workl = 20*MIN(m,n);
    uint64_t clock_start = mach_absolute_time();
    __CLPK_doublereal *lapack_work = (__CLPK_doublereal*)malloc(lapack_workl*sizeof(__CLPK_doublereal));
//    double MemoryAllocation_t = subtractTimes( mach_absolute_time(), clock_start );
    char lapack_param[1] = {'A'}; //the first min(m,n) rows of V**T (the right singular vectors) are returned in the array VT;
    char lapack_param1[1] = {'n'};
    
    clock_start = mach_absolute_time();
    dgesvd_(lapack_param, lapack_param1, &m, &n, Pts, &m, S, U, &m, NULL, &n, lapack_work, &lapack_workl, &info);
//    double dgesvd_t = subtractTimes( mach_absolute_time(), clock_start );
    
    // Handle error conditions
    if (info)
        printf("Could not compute SVD with error %d\n", info);
    else    {
        /*        printf("\n Solution is:\n");
         for( unsigned int k = 0; k<NUM_VARIABLES; k++ )
         printf("%f,", S[k]);
         printf("\n");*/
    }
    
    free( lapack_work );
    
	return info;
}






//
// MultiSVDOnSetOfPoints
//
// IN: params pointint to a structure containing the following fields:
//
//  dataPts     : ANNpointArray with all the points
//  nDims       : dimension of the ambient space, number of coordinates of each point
//  nnIdxs      : array of length nnIdxsLen of center points, with the k-th pointer pointing to a list of nnIdxsWidth[k] nearest neighbors of the k-th point
//  maxIdxsWidth: maximum number of nearest neighbors of a point, i.e. maximum of nnIdxsWidth
//  nThreads     : how many threads to use for this computation
//
// OUT:
//  return      : as returned by Cov_SVD if single threaded (in particular, 0 is success)
//  PointsS     : nDims*nnIdxsLen matrix of singular values at the nnIdxsLen points requested
 // Call parallel SVD computation at the points of the net

//Ronak ADDED:    MultiSVDOnSetOfPoints( &params_s );  is only added in Net.cpp and it is not used for our algorithm                                                                     

//
int MultiSVDOnSetOfPoints( param_MSVD_single *params )
{
    int info = 0;
    
  //  double *MeanPt=new double[params->nDims];
    
    if ( params->nThreads==1 ) {                                                                                            // Go serial (only one thread)
        unsigned long int k,p;
        
        double *PointsForSVD = new double[params->nDims * params->maxIdxsWidth];                                            // Memory allocation
        double *PointsV      = new double[params->nDims * params->nnIdxsLen * params->nDims];
        
        for ( k = 0; k<params->nnIdxsLen; k++) {                                                                            // Loop through the points and compute SVDs
            for ( p = 0; p < params->nnIdxsWidth[k]; p++)                                                                   // Organize points in local neighborhood into matrix
                memcpy(&(PointsForSVD[params->nDims*p]), params->dataPts[params->nnIdxs[k][p]], sizeof(double)*params->nDims);
            params->timings->startClock("Cov_SVD");
            info = Cov_SVD(PointsForSVD, (params->nnIdxsWidth)[k], params->nDims, params->S + k * params->nDims, PointsV);  // Compute SVD of local neighborhood
            params->timings->endClock("Cov_SVD");
        }
        delete [] PointsV;
        delete [] PointsForSVD;
    } else {                                                                                                                // Multi-threaded version
		unsigned long k, p, pts_per_thread;
        long pts_remaining;
		pthread_t threads[params->nThreads];
		param_MSVD_single params_mt[params->nThreads];
        int exit_value[params->nThreads];
        char  clockname[20];
        
		pts_per_thread = MAX(1,(unsigned long int)(ceil((double)(params->nnIdxsLen)/(double)(params->nThreads))));          // How many pts should be assigned to each thread
        pts_remaining  = params->nnIdxsLen;
        
		for (k = 0; k < params->nThreads; k++) {                                                                            // Create thread parameters and prepare threads
            if( (!MULTISCALESVDONSETOFPOINTS_AVOIDMEMORYCLASHES) || (k==0) ) {
                params_mt[k].dataPts    = params->dataPts;
            } else {
                params_mt[k].dataPts        = new ANNpoint[params->nPts];
                for( p=0; p < params->nPts; p++) {
                    params_mt[k].dataPts[p] = new ANNcoord[params->nDims];
                    memcpy(params_mt[k].dataPts[p], params->dataPts[p],params->nDims * sizeof(ANNcoord));  //size of datatype of ANNcoord
                }
            }
            params_mt[k].nDims          = params->nDims;
            params_mt[k].nPts           = params->nPts;
            params_mt[k].nnIdxsLen      = MIN( pts_per_thread, pts_remaining );
            params_mt[k].nnIdxs         = params->nnIdxs+k*pts_per_thread;
            params_mt[k].ChosenPtIdxs   = params->ChosenPtIdxs+k*pts_per_thread;
            params_mt[k].nnIdxsWidth    = params->nnIdxsWidth+k*pts_per_thread;
            params_mt[k].maxIdxsWidth   = params->maxIdxsWidth;
            params_mt[k].nThreads       = 1;
            params_mt[k].S              = params->S+k*pts_per_thread*params->nDims;
            params_mt[k].id             = k;
            
            sprintf(clockname,"Thread %lu",k);
            //params->timings->startClock( clockname );
		//	pthread_create( threads+k, NULL, MultiSVDOnSetOfPoints, (void *) (params_mt + k) );                       // Start the k-th thread
            
            pts_remaining -= params_mt[k].nnIdxsLen;
		}
        
		for (k = 0; k < params->nThreads; k++) {                                                                            // Wait for threads
			pthread_join(threads[k], (void**) (&(exit_value[k])));
            sprintf(clockname,"Thread %lu",k);
            params->timings->endClock( clockname );
            
            if( MULTISCALESVDONSETOFPOINTS_AVOIDMEMORYCLASHES && (k>0) ) {                                                                                                     // Clean up
                for( p=0; p < params->nPts; p++) {
                    delete [] params_mt[k].dataPts[p];
                }
                delete [] params_mt[k].dataPts;
            }
		}
    }
    
    return info;
}

void *MultiSVDOnSetOfPoints_multi(void *param) {                                                                            // Driver for thread function for svd calculation
    
	MultiSVDOnSetOfPoints( (param_MSVD_single *) param );
    
	return 0;
}












unsigned long int max(double* array, unsigned long int len)
{
	unsigned long int lMaxIdx = 0;
	double lMaxElement = array[0];
    
	for(register unsigned long int k=1; k<len; k++)
	{
		if( array[k]>lMaxElement )
		{
            lMaxIdx = k;
            lMaxElement = array[k];
		}
	}
    
	return lMaxIdx;
}
