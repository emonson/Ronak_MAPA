//============================================================================
// Name        : GeometricMultiResolutionAnalysis
// Author      : Mauro Maggioni
// Version     :
// Copyright   : (c) Mauro Maggioni
// Description : Fast code for producing graphs from data
//============================================================================

#include <Accelerate/Accelerate.h>          // For BLAS and LAPACK
#include "cs.h"                             // For handling sparse matrices
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <mach/mach_time.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#include "ANN_utils.h"
#include "cs_ioutils.h"
#include "Net.h"
#include "SVD.h"
#include "TimeUtils.h"
#include <vector>

using namespace std;
// Some parameters
#define NUM_THREADS_MSVD            1
#define DOWNSAMPLE_MULTISCALE_NETS  0

// Some macros
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

// Some global variables
pthread_mutex_t mutex_1         = PTHREAD_MUTEX_INITIALIZER; // Rather generic mutex
pthread_mutex_t mutex_kdtree    = PTHREAD_MUTEX_INITIALIZER; // Mutex for calls to kdtree
pthread_mutex_t mutex_arpack    = PTHREAD_MUTEX_INITIALIZER; // Mutex for calls to ARPACK

using namespace std;


// Computes multiscale SVD - multi-thread driver
typedef struct {
  ANNpointArray dataPts;
  ANNkd_tree *kdTree;
  double *PointsS;
  double *PointsV;
  param_MSVD paramMSVD;
} param_MSVD_multi;


int ComputeGraph(ANNpointArray dataPts, ANNkd_tree* kdTree, unsigned long int nPts, unsigned long int kNN, cs **A, unsigned long int kNNAutotune);      // Computes graph from point cloud
int ComputeEig(cs*T, int nev, double **Evecs, double **Evals);                                                                                          // Computes eigenvectors and eigenvalues of sparse matrix
int Cov_SVD(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *V);                                                                 // Computes singular values of covariance matrices
int Cov_SVD_withV(double* Pts, unsigned long nPts, unsigned long nDim, double *S, double *V);                 // Computes singular values and vector of covariance matrices
int Cov_SVD_withU(double* Pts, unsigned long nPts, unsigned long nDim,  double *S, double *U);                 // Computes singular values and vector of U

int ComputeMSVD(ANNpointArray dataPts, ANNkd_tree *kdTree, param_MSVD *params, double **PointsS, double **PointsV);
void *ComputeMSVD_multi(void *param);                                                                                                                   // Multi-threaded multi-svd driver

void WriteVector(const double*v, unsigned long int len);                                                                                                // Simple I/O utils
void WriteMatrix(const double*v, unsigned long int m, unsigned long int n);
int WriteMatrixToFile(const double*v, unsigned long int m, unsigned long int n, const char*FileName);
int Write3MatrixToFile(double*v, unsigned long int m, unsigned long int n, unsigned long int p, const char*FileName);
int MSVD_alloc(param_MSVD *params, double **PointsS);
int MSVD_allocU(double **PointsS, double **PointsU, unsigned long nrow_point, unsigned long ncoloumn_bags);
void EstimateDimFromSpectra (unsigned long nDims, unsigned long nscales,double **PointsS, unsigned long k, unsigned long *cDeltas, int width, unsigned long DimEst, vector<unsigned long> Goodscales );
void  Numeratoralloc(double *PowerX_C, double *PowerMultiply, unsigned long npts, double *heightminus);
void Wcomputation(double **Matrix, unsigned long nPts, unsigned long bags, double **PointsSS, double **PointsU);
float compute_slope (vector<unsigned long>&s1, vector<double>&sp, int method);
int TestCallToLAPACK3(void);
float mean (vector<double>&sp, unsigned long j, int width, unsigned long p, double **PointS, unsigned long nscales, unsigned long Dims, unsigned long k);
// Some global variables
double **T;
int tlen;
cs *s_A;
double *f_A;

string pFileName = "SampleMatrix_01.txt";

// Timing information
typedef struct {
  uint64_t kdTreeConstruction_start;
  double kdTreeConstruction;
  uint64_t GraphConstruction_start;
  double GraphConstruction;
  double EigComputation;
  double NNsearchInMSVD;
  double SVDsInMSVD;
  uint64_t TotalMSVD_start;
  double TotalMSVD;
} TimingInfo;

TimingInfo timings;

// Full matrix vector multiplication passed to ARPACK
void f_Av(int n, double *in, double*out) {
  // Set to 0 the incoming vector
  for (int i = 0; i < n; i++)
    out[i] = 0;
    
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      out[i] += in[j] * f_A[i * n + j];
}

// Sparse matrix vector multiplication passed to ARPACK
void s_Av(int n, double *in, double *out) {
  // Set to 0 the incoming vector
  for (int i = 0; i < n; i++)
    out[i] = 0;
    
  cs_di_gaxpy(s_A, in, out);
}

int main(int argc, char **argv) {
  // Parameters
  unsigned long int pK = 10; // number of nearest neighbors
  unsigned int pNNSelfTuning = 5; // autotuning nearest neighbor
    
  // Data for nearest neighbor searches
  unsigned long int nPts, nDims; // actual number of data points
  ANNpointArray dataPts; // data points
  ANNkd_tree *kdTree; // search structure
    
  cs * T;
    
  TimeList timingList;
    
  char a1[]="./Test-sample-ann";
  char a2[]="-df";
  char a3[]="/Users/hhoushiar/Downloads/Programming/CMAPA/GeometricMultiResolutionAnalysis/Sphere_528.pts";
  char *sample_array[] = {a1,a2,a3};
    
    
  // First of all load the data
  ParseInputsAndLoadData(3, sample_array, &nPts, &nDims, &dataPts);
    
    
    
  // Display some info
  cout << "\n Data set: " << nPts << " points in " << nDims << " dimensions.\n";
    
  if (pK > nPts) pK = nPts;
    
  //
  // Construct the tree for nearest neighbor searches
  //
  cout << "\n Constructing nearest neighbor data structure...";
  timingList.startClock("kdTreeConstruction");
  kdTree = new ANNkd_tree(dataPts, (int) nPts, (int) nDims); // dimension of space
  timingList.endClock("kdTreeConstruction");
  cout << "done.";
    
  //
  // Construct graph on point cloud and compute standard operators on it
  //
  cout << "\n Constructing graph...";
  timingList.startClock("GraphConstruction");
  ComputeGraph(dataPts, kdTree, nPts, pK, &T, pNNSelfTuning);
  timingList.endClock("GraphConstruction");
  cout << "done.";
    
    
  cout << "\n Computing multiscale singular values...";
  flush(cout);
    
  double *PointsS=0;
  double *PointsV=0;
  unsigned long int param_NumberOfScales = 10;
    
  timingList.startClock("TotalMSVD");                                                                                 // MM:TBD: unclear to me if the memory for this string needs to be freed
    
  //MultiscaleSVD *mSVD = 0;
    
  if (NUM_THREADS_MSVD == 1) {                                                                                        // Single-threaded version
    // Only one thread...
    param_MSVD param;
        
    param.nPts = nPts;
    param.nDims = nDims;
    param.NumberOfScales = param_NumberOfScales;
    param.RadiusMode = false;
    param.startIdx = 0;
    param.endIdx = nPts - 1;
    param.spaceAllocated = false;
        
        
    ComputeMSVD(dataPts, kdTree, &param, &PointsS, &PointsV);// multi scale singular value
  }
  timingList.endClock("TotalMSVD");
    
  // Output to file the singular values at all points and scales
  cout << "\n Output to file...";
  flush(cout);
    
  {
    Write3MatrixToFile(PointsS, nPts, param_NumberOfScales, nDims, "GeometricMultiResolutionAnalysis.out");
  }
  cout << "done.\n";
    
    
    
  // Free up memory used
  //	delete mSVD;
  delete[] PointsS;
  delete[] PointsV;
  delete kdTree;
  annClose(); // done with ANN
  cs_free(T);
    
  // Display timing info
  cout << "\nTiming Report:";
  cout << timingList;
    
  return 0;
    
}

// Ronak added finding nearest neighbours for one scale


//
// ComputeGraph
//
int ComputeGraph(ANNpointArray dataPts, ANNkd_tree* kdTree, unsigned long int nPts, unsigned long int kNN, cs **A, unsigned long int kNNAutotune) {
    
  ANNidxArray nnIdx; // near neighbor indices
  ANNdistArray distssq; // near neighbor distances
  double *sigma = new double[nPts];
    
  // Allocate memory for nearest neighbor computations
  nnIdx   = new ANNidx[kNN];
  distssq = new ANNdist[kNN];
    
  cs *Wo = cs_spalloc((int)nPts, (int)nPts, (int)kNN *(int)nPts, 1, 1);                                                                                  // Allocate memory for matrix of weights /* allocate a sparse matrix (triplet form or compressed-column form) */
  cs *W, *WT;
    
  // Compute all k-nearest neighbors
  for (unsigned int k = 0; k < nPts; k++) {
    kdTree->annkSearch(dataPts[k],(int) kNN, nnIdx, distssq, 1e-8);                                                                 // Find the kNN nearest neighbors of each point //
        
        
        
    sigma[k] = distssq[kNNAutotune];       // Fill in the matrix of weights
    for (unsigned int i = 0; i < kNN; i++) {
      cs_entry(Wo, k, nnIdx[i], exp(-distssq[i] / sigma[k]));  ///* cs_entry add an entry to a triplet matrix; return 1 if ok, 0 otherwise */
    }
  }
  delete[] nnIdx;                                                                                                                     // Free objects related to ANN searches
  delete[] distssq;
    
  // Compress sparse matrix
  W = cs_compress(Wo);
  cs_free(Wo);
    
  // Construct the sparse matrix of weights
  WT = cs_transpose(W, 1);
  *A = cs_multiply(W, WT);
  cs_symminvsqrtdeg(*A);
    
  // Free up memory
  cs_free(W);
  cs_free(WT);
    
  return 0;
}


//
// Compute <nev> eigenvalues and eigenvectors of <T>
//
int ComputeEig(cs*T, int nev, double **Evecs, double **Evals) {
  int n;
    
  n = T->m; // The order of the matrix
    
  // Allocate memory for eigenvalues and eigenvectors
  *Evals = new double[nev];
    
  // This is the global matrix that will be used for matrix-vector multiplications
  s_A = T;
    
  return 0;
}






void WriteVector(const double*v, unsigned long int len) {
  unsigned long i;
    
  for (i = 0; i < len - 1; i++)
    cout << v[i] << ",";
  cout << v[i];
}

void WriteMatrix(const double*v, unsigned long int m, unsigned long int n) {
  unsigned long i, j;
    
  for (i = 0; i < m; i++) {
    for (j = 0; j < n - 1; j++) {
      cout << v[i * n + j] << ",";
    }
    cout << v[i * n + j] << "\n";
  }
}

int WriteMatrixToFile(const double*v, unsigned long int m, unsigned long int n, const char*FileName) {
  FILE *File;
    
  File = fopen(FileName, "wt");
  if (File == NULL) return 1;
    
  fwrite(&m, sizeof(unsigned long int), 1, File);
  fwrite(&n, sizeof(unsigned long int), 1, File);
  fwrite(v, sizeof(double), m * n, File);
    
  fclose(File);
    
  return 0;
}

int Write3MatrixToFile(double*v, unsigned long int m, unsigned long int n, unsigned long int p, const char*FileName) {
  FILE *File;
    
  File = fopen(FileName, "wt");
  if (File == NULL) return 1;
    
  fwrite(&m, sizeof(unsigned long int), 1, File);
  fwrite(&n, sizeof(unsigned long int), 1, File);
  fwrite(&p, sizeof(unsigned long int), 1, File);
    
  fwrite(v, sizeof(double), m * n * p, File);
    
  fclose(File);
    
  return 0;
}
int MSVD_alloc(param_MSVD *params, double **PointsS, double **PointsV) {
  unsigned long nPtsToCompute = params->endIdx - params->startIdx + 1;
    
  if (!params->spaceAllocated) {
    // Allocate memory for singular values and vectors
    *PointsS = new double[nPtsToCompute * params->NumberOfScales * params->nDims];
    *PointsV = new double[nPtsToCompute * params->NumberOfScales * params->nDims*params->nDims];
        
    for (unsigned long int k = 0; k < nPtsToCompute * params->NumberOfScales * params->nDims; k++)
      {*(*PointsS + k) = 0;
      }
        
    for (unsigned long int k = 0; k < nPtsToCompute * params->NumberOfScales * params->nDims*params->nDims; k++)
      {*(*PointsV + k) = 0;
      }
        
    // Turn on the memory allocated flag
    params->spaceAllocated = true;
  }
    
  return 0;
}





int MSVD_allocU(double **PointsS, double **PointsU, unsigned long nrow_point, unsigned long ncoloumn_bags) {
  unsigned long nPtsToCompute = nrow_point;
    
  // Allocate memory for singular values and vectors
  *PointsS = new double[nPtsToCompute*ncoloumn_bags];
    
  PointsU= new double*[ncoloumn_bags];
  for (unsigned long i=0; i<ncoloumn_bags; i++)
    PointsU[i]= new double [ncoloumn_bags];
    
  for (unsigned long int k = 0; k < nPtsToCompute*ncoloumn_bags; k++)
    *(*PointsS + k) = 0;
  for (unsigned long i=0; i<ncoloumn_bags; i++)
    for (unsigned long j=0; j<ncoloumn_bags; j++)
      PointsU[i][j]=0;
  return 0;
}




void  Numeratoralloc(double *PowerX_C, double *PowerMultiply, unsigned long npts, double *heightminus)
{
    
    
  PowerX_C = new double[npts];
  PowerMultiply= new double [npts];
  heightminus= new double [npts];
    
    
}


void Wcomputation(double **Matrix, unsigned long nPts, unsigned long bags, double **PointsSS, double **PointsU)
{
    
  double *AT_one= new double [bags];

  double *degreesMatrix= new double [nPts];
    
  *PointsSS = new double[nPts * bags];
    
  for (unsigned long int k = 0; k < nPts * bags; k++)
    {*(*PointsSS + k) = 0;}
    
  PointsU= new double* [nPts];
  for (unsigned long i=0; i<nPts; i++)
    PointsU[i]= new double[nPts];
    
    
  for (unsigned long i=0; i<nPts;i++)
    for (unsigned long j=0; j<nPts; j++)
      { PointsU[i][j]=0;

      }
    
    
  unsigned long KMeans = 3;

  double sum;
    
  for (unsigned long j=0; j<bags;j++ )
    {
      sum=0;
        
      for (unsigned long i=0; i<nPts; i++)
            
	sum+=Matrix[i][j];
      AT_one[j]=sum;
      cout<<"\n it is correct for computing U"<<endl;
        
    }
    
  double sum2;
  for (unsigned long i=0; i<nPts; i++)
    {sum2=0;
      for (unsigned long j=0; j<bags; j++)
	sum2=Matrix[i][j]*AT_one[j]+sum2;
      degreesMatrix[i]=sum2;
    }
    
  for (unsigned long i=0; i<nPts; i++)
    if (degreesMatrix[i]==0) degreesMatrix[i]=1;
    
    
  for (unsigned long i=0; i<nPts; i++)
    degreesMatrix[i]=1/sqrtf(degreesMatrix[i]);
    
  for (unsigned long i=0; i<nPts; i++)
    for (unsigned long j=0; j<bags; j++)
      Matrix[i][j]  = Matrix[i][j]*degreesMatrix[i];
    
  double *points = new double [nPts*bags];
    
  for (unsigned long i=0; i<nPts; i++)
    for (unsigned long j=0; j<bags; j++)
      points[i*bags+j]  =  Matrix[i][j] ;
    
    
  Cov_SVD_withU( points, nPts, bags, (*PointsSS),(*PointsU));
    
  for (unsigned long i=0;i<KMeans;i++){
    for(int j=0;j<bags;j++){
      cout<<PointsU[i][j]<<'\t';
    }
    cout<<endl;
  }
    
    
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
void EstimateDimFromSpectra (unsigned long nDims, unsigned long nscales,double **PointsS, unsigned long k,unsigned long *cDeltas, int width, unsigned long DimEst, vector<unsigned long> Goodscales )
		
{
  //Goodscales= vector<unsigned long>(2, std::numeric_limits<float>::quiet_NaN());
  float alpha1=0.1, alpha0=0.2;
  DimEst= nDims;

  int iMax=12;
  unsigned long jMax= nscales;
  unsigned long p=nDims;
  vector<double> sp ;

  for (unsigned long iscale=0; iscale<nscales; iscale++)
    sp.push_back (*((*PointsS) + k * nscales * nDims + iscale * nDims + nDims));
  int j;
  int i=width; 
  float slope;
  vector<unsigned long> s1; 
  vector<double> spp;
  for (int a=0; a<i; a++)
    {
      s1.push_back(cDeltas[a]);
      spp.push_back(sp[a]);
    }

	
  slope= compute_slope (s1,spp, 2 );
  ///// empty current s1 and spp ////
  while (!s1.empty())
    {
      s1.pop_back();
    }

  while (!spp.empty())
    {
      spp.pop_back();
    }
  /////////////////////////////////////////
		

  while (i<iMax && slope>0.1)
    {
      i++;
      for (int a=i-width; a<i; a++)
	{
	  s1.push_back(cDeltas[a]);
	  spp.push_back(sp[a]);
	}
      slope= compute_slope (s1, spp, 2);
	
      ///// empty current s1 and spp ////
      while (!s1.empty())
	{
	  s1.pop_back();
	}

      while (!spp.empty())
	{
	  spp.pop_back();
	}

    }
  char isaNoisySingularValue;
  if (i>iMax) 
    {
      isaNoisySingularValue = false;
        
      ///// empty current s1 and spp ////
      while (!s1.empty())
	{
	  s1.pop_back();
	}
        
      while (!spp.empty())
	{
	  spp.pop_back();
	}
        
      /////////////////////////////
    }
  else
    {
	
      j=i; 
      while (j<jMax-1 && slope<=alpha0 )
	{
	  j++;
	  for (int a=j-width; a<j; a++)
	    {
	      s1.push_back(cDeltas[a]);
	      spp.push_back(sp[a]);
	    }

	  slope=compute_slope(s1, spp, 2);
	  ///// empty current s1 and spp ////
	  while (!s1.empty())
	    {
	      s1.pop_back();
	    }

	  while (!spp.empty())
	    {
	      spp.pop_back();
	    }
	}
      j=j-1;
      Goodscales[0]=i-1;
      Goodscales[1]=j;
      isaNoisySingularValue = true;
    }

  ///// empty current s1 and spp ////
  while (!s1.empty())
    {
      s1.pop_back();
    }

  while (!spp.empty())
    {
      spp.pop_back();
    }			
  /////////////////////////////

  while (p>1 && isaNoisySingularValue)
    {
      p=p-1; 
      while (!sp.empty())
	{
	  sp.pop_back();
	}
      for (unsigned long int iscale=0; iscale<nscales; iscale++)
	sp.push_back (*((*PointsS) + k * nscales * nDims + iscale * nDims + p-1));
      // % find a lower bound for the optimal scale
      for (int a=0; a<i; a++)
	{
	  s1.push_back(cDeltas[a]);
	  spp.push_back(sp[a]);
	}
      i=width; slope=compute_slope(s1, spp, 2);
      ///// empty current s1 and spp ////
      while (!s1.empty())
	{
	  s1.pop_back();
	}

      while (!spp.empty())
	{
	  spp.pop_back();
	}			

      while(i<iMax && slope>0.1)
	{
	  i++; 
	  for (int a=i-width; a<i; a++)
	    {
	      s1.push_back(cDeltas[a]);
	      spp.push_back(sp[a]);
	    }
	  slope=compute_slope(s1, spp, 2);
	  ///// empty current s1 and spp ////
	  while (!s1.empty())
	    {
	      s1.pop_back();
	    }

	  while (!spp.empty())
	    {
	      spp.pop_back();
	    }		
	}
      //   % find an upper bound for the optimal scale
      if (slope<=alpha0)
	{
	  j=i;
	  while (j<jMax && slope<=alpha0)
	    {
	      j++;
	      for (int a=j-width; a<j; a++)
		{
		  s1.push_back(cDeltas[a]);
		  spp.push_back(sp[a]);
		}
		
	      slope= compute_slope(s1,spp, 2);
	      ///// empty current s1 and spp ////
	      while (!s1.empty())
		{
		  s1.pop_back();
		}

	      while (!spp.empty())
		{
		  spp.pop_back();
		}	

	    }
	  j--;
		
	  if (mean (sp,j, width, p, PointsS, nscales, nDims, k )>0.2)
	    isaNoisySingularValue = false;
	  else 
	    {
	      Goodscales[0]=i-1;
	      Goodscales[1]=j;
	    } 
	
	}

      else 
	{
	  isaNoisySingularValue = false;

	}
	
    }
  DimEst = p;
}

/////////////////////slope//////////////////////////
float compute_slope (vector<unsigned long>&s1, vector<double>&sp, int method)
{
  float slope;
  double sum=0, sum_sp=0;
  unsigned long sum_s1=0, sum_pow=0; 
	
  if(s1.empty())
    {
      cout << "No values" << endl;
      return 0;
    }
  if(sp.empty())
    {
      cout << "No singular values" << endl;
      return 0;
    }
    
  double M;
  for (int i=0; i<s1.size(); i++)
    {	
      M=s1[i]*sp[i];
      sum+=M;
    }
  for (int i=0; i<s1.size(); i++)
    sum_s1+=s1[i];

  for (int i=0; i<sp.size(); i++)
    sum_sp+=sp[i];
	
  for (int i=0; i<s1.size(); i++)
    {
		
      sum_pow+= powf(s1[i], 2);
    }
  slope= (sum-sum_s1*sum_sp/s1.size())/(sum_pow-powf(sum_s1,2)*s1.size());

  return (slope);
}

/////////////////////////compute mean//////////////////////////////
float mean (vector<double>&sp, unsigned long j, int width, unsigned long p, double **PointS, unsigned long nscales,unsigned long Dims, unsigned long k)
{
	
  vector<vector<double>> S_MSVD;//[atscale][jth singular]
  S_MSVD=vector<vector<double>>(nscales, vector<double>(Dims, std::numeric_limits<double>::quiet_NaN()));
  // last singular value for scales
  for (unsigned long int iscale=0; iscale<nscales; iscale++)
    for (unsigned long int singular=0; singular<Dims; singular++)
      {
	S_MSVD[iscale][singular]= *((*PointS) + k * nscales * Dims + iscale * Dims + singular);
            
      }
  float Mean_t=0;
  vector <double>S_MSVD2;
  vector<double>S_MSVD3;
  vector<double>S_MSVD4;
  for (unsigned long  int a=j-width; a<j; a++)
    {
      S_MSVD2.push_back(S_MSVD[a][p+1]);
    }

  for (unsigned long  int a=j-width; a<j; a++)
    {
      S_MSVD3.push_back(S_MSVD[a][1]);
    }

  for (unsigned long  int a=0; a<S_MSVD3.size(); a++)
    {
      S_MSVD4.push_back(S_MSVD2[a]/S_MSVD3[a]);
    }
		
  vector <double>spp;

  for (unsigned long a=j-width; a<j; a++)
    {
      spp.push_back(sp[a]);
    }
  vector <double> minus;
  for (unsigned long  int a=0; a<S_MSVD4.size(); a++)
    {
      minus.push_back(spp[a]-S_MSVD4[a]-S_MSVD2[a]);
    }
  for (unsigned long  int a=0; a<minus.size(); a++)
    {
      Mean_t+= minus[a];
    }
  Mean_t=Mean_t/minus.size();

  return (Mean_t);

}

//
// int ComputeMSVD(ANNpointArray dataPts, ANNkd_tree *kdTree, param_MSVD *params, double **PointsS, double **PointsV)
//
// Computes svd on
//
//
//	IN:
//	dataPts	 			: data set
//	nPts,nDims			: number of points and their dimension
//	kdTree 				: tree for nearest neighbor searches
//	param_RadiusMode  	: go multiscale in radii or nearest neighbors
//	param_NumberOfScales: number of scales to compute
//
//
//	OUT:
//	PointsS				: PointsS[k][j][i] is the i-th singular value at scale j for point k
//  PointsV				: MM: NOT IMPLEMENTED YET
//
// Computes multiscale SVD - single threaded
//
int ComputeMSVD(ANNpointArray dataPts, ANNkd_tree *kdTree, param_MSVD *params, double **PointsS, double **PointsV) {
  // Parameters
	
  // Variables
  unsigned long int *kNNs = 0;
  unsigned long int maxNNs = 0;
  //unsigned long int width=5; // steps for estimating the dimensions linear or not
  unsigned long int k;
  double sum, sum1;
    
  unsigned long int nPtsToCompute;// Number of points for which the computation will be done
  unsigned long int  kCurPt;
  ANNidxArray nnIdx; // Near neighbor indices
  ANNdistArray dists; // Near neighbor distances
  double *PointsForSVD; // Memory for local points on which to compute the SVD
  double *PointsForMean;// memory for local points on which to compute the mean after estimating dimension
    
    
  unsigned long int  j, p, d; // We are going to need indices!
  unsigned long int *nLocalPts;
  nLocalPts = new unsigned long int[params->nPts * params->NumberOfScales];
  unsigned long int highboundry;
    
  unsigned long int EstimateDimension = params->nDims;
  ANNdist *Radii = 0;
  vector<unsigned long> GoodScales(2, 100);//(nScales); 
       
    
  double** X_C;
  double** Multiply;
  double Summulti;
  unsigned long iVmulti, imulti;
    
  // Allocate memory for Radii or kNNs
  if ((params->RadiusMode)) {
    maxNNs = params->nPts;
    Radii = new ANNdist[params->NumberOfScales];
    for (k = 0; k < params->NumberOfScales; k++) {
      Radii[k] = 0.1 * (double) k; // MM: needs to be done correctly here
    }
  } else {
    kNNs = new unsigned long [params->NumberOfScales];
    //unsigned long int kNNstep;
        
    //Ronak added for mapa
    unsigned long int kNNstepmapa;
    double dmax=(double)params->nDims - 1;
        
        
    //----------------------------------------------------
    // Going up one scale increases the number of nearest neighbor by a constant number,
    // chosen so that at the largest scale we get all the points
    //------------------------------------------------------
        
    kNNstepmapa = (unsigned long int) floor(MIN(50* dmax * (double)log(dmax), (double)params->nPts/5)/ (double) params->NumberOfScales); // number of points per scale. which is actually maxKNN/opt.nScales in matlab mapa
    for (unsigned long int i = 0; i < params->NumberOfScales; i++)
      kNNs[i] = kNNstepmapa * (i + 1);
        
    maxNNs = kNNs[params->NumberOfScales - 1];
  }
    
    
  // Allocate memory for all the singular values/vectors at all scales for all points
  MSVD_alloc(params, PointsS,PointsV);
    
  nPtsToCompute = params->endIdx - params->startIdx + 1;
    
    
  double pEPS = 1e-4 ; // precision in nearest neighbor computation
    
  unsigned long *highboundry2= new unsigned long [nPtsToCompute];
  // Allocate memory for the output of nn/radius searches
  nnIdx   = new ANNidx[maxNNs];
  dists  = new ANNdist[maxNNs];
  PointsForSVD = new double[maxNNs * params->nDims];
  double *PowerX_C,*PowerMultiply;
  double *heightminus=0;
    
  // Array for how many nn/radius-neighbors are found, for every pt and radius
  nLocalPts = new unsigned long int[nPtsToCompute * params->NumberOfScales];
    
  /*--------------------------------------------------
    nPtsToComputeForX_C is the size of whole datasets used which is not neseccarily
    equal to nPtsToCompute
    //--------------------------------------------*/
    
    
  unsigned long nPtsToComputeForX_C = params->nPts;
    
  double ** CreatingA_matrix= new double* [params->nPts];
  for (unsigned long i=0; i<params->nPts; i++)
    CreatingA_matrix[i]= new double[nPtsToCompute];
    
    
    
  for (unsigned long i=0; i<params->nPts;i++)
    for (unsigned long j=0; j<nPtsToCompute; j++)
      { CreatingA_matrix[i][j]=0;
      }
    
  //loop through the points
    
  for (k = 0; k < nPtsToCompute; k++) {
    // Index of current point
    kCurPt = k + params->startIdx;
        
    cout<<"\n kCurPt"<<kCurPt<<endl;
        
    // Loop through the scales
    for (j = 0; j < params->NumberOfScales; j++)
            
      {
	pthread_mutex_lock(&mutex_1);
	kdTree->annkSearch(dataPts[kCurPt], (int)(kNNs[j]), nnIdx, dists, pEPS);                                // Find nearest neighbors
	pthread_mutex_unlock(&mutex_1);
	nLocalPts[k * params->NumberOfScales + j] = kNNs[j];                                                    // Save number of near neighbors found
			
	for (p = 0; p < nLocalPts[k * params->NumberOfScales + j]; p++) {                                           // Organize selected points into matrix.
	  for (d = 0; d < params->nDims; d++)
	    { //cout<<"\n data"<<"k:"<<k<<"j:"<<j<<"d:"<<d<<(dataPts[nnIdx[p]])[d];
	      PointsForSVD[p * params->nDims + d] = (dataPts[nnIdx[p]])[d];
	    }
	}
	// Compute the singular values/vectors of the points for this subset of points
	Cov_SVD(PointsForSVD, nLocalPts[k * params->NumberOfScales + j], params->nDims, (*PointsS) + k * params->NumberOfScales * params->nDims + j * params->nDims,(*PointsV)+k*params->NumberOfScales*params->nDims*params->nDims+j*params->nDims*params->nDims);
	for (unsigned long int i=0; i< (params->nDims); i++)
	  {
	  }
            
            
      }
    EstimateDimFromSpectra(params->nDims,params->NumberOfScales,PointsS,k, kNNs, 5, EstimateDimension, GoodScales);
    cout<<"Run shod"<<endl;
      cout<<"GoodScales[1]= "<<GoodScales[1]<<endl;
  }
    delete[] kNNs;
    
    delete[] nLocalPts;
    delete[] PointsForSVD;
    
    delete[] nnIdx;
    delete[] dists;
 
 

    return 0;
  }
