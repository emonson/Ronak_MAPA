/*
 * Net.h
 *
 *  Created on: Jul 18, 2009
 *      Author: Mauro
 */

#ifndef NET_H_
#define NET_H_

#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <pthread.h>
#include <stdint.h>
#include "ANN_utils.h"
#include "TimeUtils.h"
#include "SVD.h"

int MultiSVDOnSetOfPoints( param_MSVD_single *params );
void *MultiSVDOnSetOfPoints_multi(void *param);
int ComputeMSVD(ANNpointArray dataPts, ANNkd_tree *kdTree, param_MSVD *params, double **PointsS, double **PointsV);

typedef struct {
	unsigned long int *Vol; // Number of points in each cell
	double *Radii; // Radius of each cell
	double *S; // Singular values for cell k are at S[k*nS]
	unsigned long int nS; // How many singular values in one cell
} NETSTATS;


//
// Net
//
class Net {
    
    friend class MultiscaleSVD;
    
private:
	unsigned long int *idxs;
	NETSTATS NetStats;
    
    TimeList timings;

public:
	unsigned long int n;

	Net(unsigned long int NumberOfNetPoints=0, const unsigned long int *Idxs=NULL);
	~Net();

    void PutIdxs( const unsigned long int *Idxs, unsigned long int NumberOfNetPoints );
    void SetIdxs( unsigned long int *Idxs, unsigned long int NumberOfNetPoints );
	void PutVols( const unsigned long int *Vol);                                                                            // Copies Vol into the Vol's of NetStats
	void SetVols( unsigned long int *Vol);                                                                                  // Sets Vol into the Vol's of NetStats
	void PutRadii( const double *Radii);                                                                                    // Copies Radii into the Radii in NetStats
	void SetRadii( double *Radii);                                                                                          // Sets Radii into the Radii in NetStats
	void PutS(unsigned long int nS, const double*S, unsigned long int NS = 0);                                              // Copies S into the S of NetStats
	void SetS(unsigned long int nS, double*S, unsigned long int NS = 0);                                                    // Sets S into the S of NetStats

	void WriteS(FILE *File);
};


//
// MultiscaleSVD
//

class MultiscaleSVD {

private:
	unsigned long int N;
	unsigned long int J;

	ANNpointArray dataPts;
	ANNkd_tree *kdTree;
	Net **Nets;
    
    TimeList timings;
    
	int ComputeMSVDnetSingleScale(const param_MSVD *params, ANNdist Radius, unsigned long int kNN, unsigned long int maxNNs, Net **vNet);
public:
	pthread_mutex_t *mutex_kdtree; // Mutex for calls to kdtree

	MultiscaleSVD(unsigned long int NumberOfPoints, ANNpointArray dataPts_in, ANNkd_tree *kdTree_in );
	~MultiscaleSVD();

	int ComputeMSVDnets(param_MSVD *params);

	void SetNet(unsigned long int j, Net *net);

	void WriteS(FILE *File);
    
    void PrintTimingsReport( void );
};

#endif /* NET_H_ */
