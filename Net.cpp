/*
 * Net.cpp
 *
 *  Created on: Jul 18, 2009
 *      Author: Mauro
 */
#include <string.h>
#include <iostream>
#include <pthread.h>
#include "Net.h"

using namespace std;

extern int Cov_SVD(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *V);
extern double subtractTimes( uint64_t endTime, uint64_t startTime );

#define MAX_SINGULARVALUES 20
#define NUM_THREADS_MSVD_SVD_CALCULATION 4


// Some macros
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

unsigned long int max(double* array, unsigned long int len);


//
// Net implementation
//

Net::Net(unsigned long int NumberOfNetPoints, const unsigned long int *Idxs) {
	n = NumberOfNetPoints;
    if( Idxs!= NULL ) {
        idxs = new unsigned long int[n];
        memcpy(idxs, Idxs, n * sizeof(unsigned long int));
    }
    else {
        idxs = NULL;
    }
	NetStats.Vol = new unsigned long int[n];
	NetStats.Radii = new double[n];
	NetStats.S = 0;
}

Net::~Net() {
	delete[] NetStats.Vol;
	delete[] NetStats.Radii;
	delete[] NetStats.S;
	delete[] idxs;
}


void Net::PutIdxs( const unsigned long int *Idxs, unsigned long int NumberOfNetPoints ) {
    if( idxs!= NULL ) {
        delete [] idxs;
    }
    
    n    = NumberOfNetPoints;
    idxs = new unsigned long int[n];
    memcpy(idxs, Idxs, n*sizeof(unsigned long int));
}
void Net::SetIdxs( unsigned long int *Idxs, unsigned long int NumberOfNetPoints ) {
    if( idxs!= NULL ) {
        delete [] idxs;
    }
    idxs = Idxs;
}

void Net::PutVols(const unsigned long int *Vol) {
    if( NetStats.Vol!= NULL )
        delete [] NetStats.Vol;
    
    NetStats.Vol = new unsigned long int[n];
	memcpy(NetStats.Vol, Vol, n * sizeof(unsigned long int));
}

void Net::SetVols( unsigned long int *Vol) {
    if( NetStats.Vol!= NULL )
        delete [] NetStats.Vol;
    
    NetStats.Vol = Vol;
}


void Net::PutRadii(const double *Radii) {
    if( NetStats.Radii!= NULL )
        delete [] NetStats.Radii;
    
    NetStats.Radii = new double[n];
	memcpy(NetStats.Radii, Radii, n * sizeof(double));
}

void Net::SetRadii(double *Radii) {
    if( NetStats.Radii!= NULL )
        delete [] NetStats.Radii;
    
    NetStats.Radii = Radii;
}


void Net::PutS(unsigned long int nS, const double*S, unsigned long int NS) {
    if( NetStats.S!=NULL )
        delete [] NetStats.S;
    
	NetStats.nS = nS;
	NetStats.S = new double[n * nS];
	if ((NS == 0) || (NS == nS))
		memcpy(NetStats.S, S, n * nS * sizeof(double));
	else
		for (unsigned long int k = 0; k < n; k++)
			memcpy(NetStats.S + k * nS, S + k * NS, nS * sizeof(double));
}

void Net::SetS(unsigned long int nS, double*S, unsigned long int NS) {
    if( NetStats.S!=NULL )
        delete [] NetStats.S;
    
	NetStats.nS = nS;
	NetStats.S = S;
}


void Net::WriteS(FILE *File) {
	// Write number of points, number of singular values
	fwrite(&n, sizeof(unsigned long int), 1, File);
	fwrite(&(NetStats.nS), sizeof(unsigned long int), 1, File);
	// Write vector of indices of the points in the net
	fwrite(idxs, sizeof(unsigned long int), n, File);
	// Write the vector S
	fwrite(NetStats.S, sizeof(double), n * NetStats.nS, File);
}

//
// MultiscaleSVD implementation
//

MultiscaleSVD::MultiscaleSVD(unsigned long int NumberOfPoints, ANNpointArray dataPts_in, ANNkd_tree *kdTree_in) {
	N = NumberOfPoints;
	J = 0;
    
	dataPts = dataPts_in;
	kdTree = kdTree_in;
	mutex_kdtree = 0;
	Nets = 0; //new Net*[J];
}

MultiscaleSVD::~MultiscaleSVD() {
    
	if ( Nets!= NULL ) {
		for (unsigned long int j = 0; j < J; j++)
			delete Nets[j];
		delete Nets;
	}
}

void MultiscaleSVD::SetNet(unsigned long int j, Net *net) {
	Nets[j] = net;
}

void MultiscaleSVD::WriteS(FILE *File) {
	// Write number of points and of scales
	fwrite(&N, sizeof(unsigned long int), 1, File);
	fwrite(&J, sizeof(unsigned long int), 1, File);
    
	// Write the S in the nets at each scale
	for (unsigned long int j = 0; j < J; j++) {
		Nets[j]->WriteS(File);
	}
}

//
// ComputeMSVDnets
//
int MultiscaleSVD::ComputeMSVDnets(param_MSVD *params) {
    
	unsigned long int k, j;
    
	// Set up for multiscale nets
	if ( Nets!=NULL ) {
		for (j = 0; j < J; j++)
			delete Nets[j];
		delete Nets;
	}
	J = params->NumberOfScales;
	Nets = new Net*[J];
    
	unsigned long int maxNNs = 0;
	unsigned long int *kNNs = 0;
	ANNdist *Radii = 0;
	// Allocate memory for Radii or kNNs
	if ((params->RadiusMode)) {                                                                                 // Go by radii
		maxNNs = params->nPts;
		Radii = new ANNdist[params->NumberOfScales];
		for (k = 0; k < params->NumberOfScales; k++) {
			Radii[k] = 0.1 * (double) k;                                                                        // MM:TBD needs to be done correctly here
		}
	} else {                                                                                                    // Go by #of nearest neighbors
		kNNs = new unsigned long int[params->NumberOfScales];
		unsigned long int kNNstep;
		
		kNNstep = (unsigned long int) floor((double) params->nPts / (double) (params->NumberOfScales));         // Linearly scale number of nn's with scale
		kNNstep = floor(kNNstep / 20);
		for (j = 0; j < params->NumberOfScales; j++)
			kNNs[j] = MAX(kNNstep * (j + 1),1);
        
		maxNNs = kNNs[params->NumberOfScales - 1];
	}
	for (j = 0; j < J; j++) {                                                                                    // Loop through the scales
		if (params->RadiusMode)
			ComputeMSVDnetSingleScale(params, Radii[j], 0, maxNNs, &Nets[j]);                                    // Construct j-th net
		else
			ComputeMSVDnetSingleScale(params, 0, kNNs[j], maxNNs, &Nets[j]);
	}
    
	return 0;
}

//
// ComputeMSVDnetSingleScale
//
//  Constructs a net of well-separated points and computes local svd in the neighborhood of each point
//
// IN:
//      params  : pointer to param_MSVD structure containing various parameters, as for ComputeMSVDnets
//      Radius  : radius around each point of the net
//      kNN     : number of nearest neighbors around each point of the net
//      maxNNs  : maximum number of nearest neighbors to consider
//      vNet    : array of nets at different scales
//
//
int MultiscaleSVD::ComputeMSVDnetSingleScale(const param_MSVD *params, ANNdist Radius, unsigned long int kNN, unsigned long int maxNNs, Net **vNet) {
    
    // Input check
    if( !(params->RadiusMode) ) {
        maxNNs = min(kNN,maxNNs);
    } else {
        maxNNs = min( params->endIdx-params->startIdx+1,maxNNs );
    }
    
    // Parameters
	double pEPS = 1e-4;                                                                                         // Precision in nearest neighbor computation
    
	// Variables
	unsigned long int   k, k_subidx, p, kCurPtidx;                                                              // We are going to need indices!
	unsigned long int   nPtsToCompute = params->endIdx - params->startIdx + 1;                                  // Number of points for which the computation will be done
    unsigned long int   nPts = nPtsToCompute;
	double              *PointsForSVD = new double[maxNNs*params->nDims*nPtsToCompute];                         // Memory for local points on which to compute the SVD
	ANNidxArray*        nnIdxs = new ANNidxArray[nPtsToCompute];                                                // Near neighbor indices
	ANNdistArray        dists;                                                                                  // Near neighbor distances
    for ( k=0; k<nPtsToCompute; k++) {                                                                          // Allocate memory for the output of nn/radius searches
        nnIdxs[k] = new ANNidx[maxNNs];
    }
	dists = new ANNdist[maxNNs];
    
	unsigned long int *nLocalPts    = new unsigned long int[nPtsToCompute];                                     // Array for how many nn/radius-neighbors are found at a fixed scale
	unsigned long int *ChosenPtIdxs = new unsigned long int[nPtsToCompute];                                     // Array for the indices of the points in the net at this scale
	double *nLocalRadii             = new double[nPtsToCompute];
	bool *flagPoints                = new bool[nPtsToCompute];                                                  // Array that keeps track of which points have already been considered at each scale
	unsigned long int nPtsLeftToCompute;                                                                        // Counter for how many points are still left to consider

	for (k = 0; k < nPtsToCompute; k++)     flagPoints[k] = false;                                              // Initializations
	ANNdist RadiusSquared = Radius * Radius;
    
	// Work: Construct the points in the net
	*vNet = new Net;
    
    (*vNet)->timings.startClock("NetConstruction");
	for (k = 0, k_subidx = 0, nPtsLeftToCompute = nPtsToCompute; nPtsLeftToCompute > 0; k++, k_subidx++) {      // Loop through the points and construct a well-distributed net
		while ( flagPoints[k] )   k++;                                                                          // Find the next available point
		kCurPtidx = k + params->startIdx;                                                                       // Get its absolute index
		ChosenPtIdxs[k_subidx] = kCurPtidx;                                                                     // Save its index
		if (mutex_kdtree)   pthread_mutex_lock( mutex_kdtree );                                                 // The ANN functions are not thread-safe
        (*vNet)->timings.startClock("NetConstruction_nn");
		if ( params->RadiusMode ) {                                                                             // Find the RadiiSq[l]-neighbors of each point. Result in nnIdxs and dists.
			nLocalPts   [k_subidx]  = kdTree->annkFRSearch(dataPts[kCurPtidx], RadiusSquared, (int)maxNNs, nnIdxs[k_subidx], dists, pEPS);
			nLocalRadii [k_subidx]  = Radius;
		} else {                                                                                                // Find the kNN-neighbors of each point. Result in nnIdxs and dists.
			kdTree->annkSearch(dataPts[kCurPtidx], (int)kNN, nnIdxs[k_subidx], dists, pEPS);
			nLocalPts[k_subidx]     = kNN;
			nLocalRadii[k_subidx]   = max(dists,kNN);
		}
        (*vNet)->timings.endClock("NetConstruction_nn");
		if (mutex_kdtree)   pthread_mutex_unlock( mutex_kdtree );
		
		for (p = 0; p < nLocalPts[k_subidx]; p++)                                                               // Mark the points in the neighborhood as done
			if (!flagPoints[nnIdxs[k_subidx][p]]) {                                                 			// Check if the point was already selected
				flagPoints[nnIdxs[k_subidx][p]] = true;                                                         // Mark the point as selected
				--nPtsLeftToCompute;                                                                            // Decrease the number of points remaining
			}
	}
    (*vNet)->timings.endClock("NetConstruction");
    
	double *PointsS = new double[k_subidx * params->nDims];                                                     // Preparation for parallel SVD computation
    param_MSVD_single params_s;
    params_s.dataPts        = dataPts;
    params_s.nDims          = params->nDims;
    params_s.nPts           = nPts;
    params_s.ChosenPtIdxs   = ChosenPtIdxs;
    params_s.nnIdxs         = nnIdxs;
    params_s.nnIdxsLen      = k_subidx;
    params_s.nnIdxsWidth    = nLocalPts;
    params_s.maxIdxsWidth   = params->RadiusMode ? maxNNs : kNN;
    params_s.nThreads       = NUM_THREADS_MSVD_SVD_CALCULATION;
    params_s.S              = PointsS;
    
    (*vNet)->timings.startClock("SVDComputation");
    MultiSVDOnSetOfPoints( &params_s );                                                                         // Call parallel SVD computation at the points of the net
    (*vNet)->timings.endClock("SVDComputation");
    
    (*vNet)->SetIdxs(ChosenPtIdxs,k_subidx);
	(*vNet)->SetVols(nLocalPts);
	(*vNet)->SetRadii(nLocalRadii);
	(*vNet)->SetS(MIN(MAX_SINGULARVALUES, params->nDims), PointsS, params->nDims);
    
        	
	delete[] flagPoints;
	delete[] PointsForSVD;
    for ( k=0; k<nPtsToCompute; k++) { delete[] nnIdxs[k];  }
    delete[] nnIdxs;
	delete[] dists;
    
	return 0;
}

void MultiscaleSVD::PrintTimingsReport( void )
{
    for( unsigned long int j=0; j<J; j++ )
        cout << "\n Timing for net at scale " << j <<": " << Nets[j]->timings;
    
    flush( cout );
    return;
}



