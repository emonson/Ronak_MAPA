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
// #include "gromacs/gmx_arpack.h"
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
//int MSVD_alloc_onlyV(param_MSVD *params, double **PointsV);
//int MSVD_alloc_2(param_MSVD *params,double **Multiply, double highboundry, unsigned long k , double **X_C, unsigned long int nnpts);
int MSVD_allocU(double **PointsS, double **PointsU, unsigned long nrow_point, unsigned long ncoloumn_bags);
//int MSVD_allocV(double **PointsV, unsigned long nDims, unsigned long nPts, unsigned long nPtsToCompute);
float compute_slope(unsigned long int *ScaleArray,vector<vector<double>> &Singular_MSVD, int method, unsigned long int width,unsigned long int start ,unsigned long int end, unsigned long int iscales, unsigned long int p);
float compute_Mean(unsigned long int *ScaleArray,vector<vector<double>> &Singular_MSVD, unsigned long int width,unsigned long int start ,unsigned long int end,unsigned long int iscales,unsigned long int p);
// Ronak added
void EstimateDimFromSpectra( unsigned long int Dims, unsigned long int nscales, double **PointsS,unsigned long int j, unsigned long int k, unsigned long int *s1,unsigned long int  width, unsigned long int DimEst);

//float compute_Mean(unsigned long int *ScaleArray,vector<vector<double>> &Singular_MSVD, unsigned long int width, unsigned long int l, unsigned long p);
//int Compute_MSVD_OptimalScale(ANNpointArray dataPts, ANNkd_tree *kdTree, param_MSVD *params, double **PointsS, double **PointsV, unsigned long int Jstart, unsigned long int maxNNs, unsigned long int *kNNs, int width, unsigned long int *nLocalPts,ANNdistArray dists, ANNidxArray nnIdx, double *PointsForSVD);
void  Numeratoralloc(double *PowerX_C, double *PowerMultiply, unsigned long npts, double *heightminus);
void Wcomputation(double **Matrix, unsigned long nPts, unsigned long bags, double **PointsSS, double **PointsU);

int TestCallToLAPACK3(void);

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
    
    // Do A*in+out and store the result in out
    cs_di_gaxpy(s_A, in, out);
    /*for(int i=0; i<n;i++)
     cout << out[i] << ",";
     cout << "\n";*/
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
    //char a4[]="-qf";
    // char a5[]="/Users/ronak/Programming/CMAPA/ann_1.1.1 2/sample/query.pts";
    //char a2,a3,a4,a5;
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
            //cout<<"\n test test S:"<<*(*PointsS + k)<<"k="<<k<<endl;
        }
        
        for (unsigned long int k = 0; k < nPtsToCompute * params->NumberOfScales * params->nDims*params->nDims; k++)
        {*(*PointsV + k) = 0;
            //cout<<"\n test test V:"<<*(*PointsV + k)<<"k="<<k<<endl;
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
    //    V=new double*[NROW];
    //  for(int i=0;i<NROW;i++)
    //   *V=new double[NCOL];*/
    
    // ronak added:
    // *PointsU = new double[ncoloumn_bags*ncoloumn_bags];
    
    for (unsigned long int k = 0; k < nPtsToCompute*ncoloumn_bags; k++)
        *(*PointsS + k) = 0;
    for (unsigned long i=0; i<ncoloumn_bags; i++)
        for (unsigned long j=0; j<ncoloumn_bags; j++)
            PointsU[i][j]=0;
    
    
    
    // Turn on the memory allocated flag
    //	params->spaceAllocated = true;
	
    
    return 0;
}



/////Ronak added//////////////////////////////////////////

void EstimateDimFromSpectra( unsigned long int Dims,unsigned long int nscales, double **PointsS, unsigned long int k, unsigned long int *s1, unsigned long int  width, unsigned long int &DimEst, unsigned long int *Goodscales)
{ float alpha1=0.1, alpha2=0.2;
    
    //int modcheck = j % (width-1);
    
    unsigned long int  iMax= MIN(12, nscales);
    unsigned long int jMax= nscales;
    
    // double Singular_MSVD[nscales]; // array of last signular values
    
    
    unsigned long int *ScaleArray;
    ScaleArray = new unsigned long int[nscales]; // array of scales
    
    char isaNoisySigularValue;
    unsigned long int p;
    p = Dims-1;
    
    vector<vector<double>> Singular_MSVD;//[atscale][jth singular]
    Singular_MSVD= vector<vector<double>>(nscales, vector<double>(Dims, std::numeric_limits<double>::quiet_NaN()));
    
    // last singular value for scales
    for (unsigned long int iscale=0; iscale<nscales; iscale++)
        for (unsigned long int singular=0; singular<Dims; singular++)
        {
            Singular_MSVD[iscale][singular]= *((*PointsS) + k * nscales * Dims + iscale * Dims + singular);
            
        }
    cout<<"ScaleArray: scales array"<<endl;
    // ScaleArray: scales array
    for (int i=0;i<nscales; i++)
    {
        ScaleArray[i]= s1[i];
    }
    
    
    
    float slope =0;
    int i;
    unsigned long int l, l2;
    
    i= width-1;
    unsigned long int i2;
    slope = compute_slope(ScaleArray, Singular_MSVD, 2, width, 0, i , nscales, p);//scalearray from 0 to width
    cout<<slope<<"  1"<<endl;
    cout<<i<<endl;
    cout<<iMax<<endl;
    while ((i<iMax) && (slope>alpha1))
    {
        i++;
        
        i2=i-width+1;
        cout<<"i2= "<<i2<<endl;
        slope= compute_slope(ScaleArray,Singular_MSVD, 2, width, i2, i, nscales,p );//scalearray from 1 to width+1
        cout<<slope<<"->"<<i<<endl;
    }
    if (i>=iMax)
        isaNoisySigularValue=false;
    else
    {
        l=i;
        while ((l<jMax-1)&& (slope<=alpha2))
        {
            l++;
            l2=l-width+1;
            slope= compute_slope(ScaleArray,Singular_MSVD, 2, width,l2, l,nscales, p );
        }
        
        l=l-1;
        Goodscales[0]=i-1;
        Goodscales[1]=l;
        isaNoisySigularValue=true;
        
    }
    ////// next step
    //    float meanvalue;
    
    while (p>1 && (isaNoisySigularValue==true))
    {
        p=p-1;// next smallest singular value for scales
        
        
        
        ///find the lower bound
        i=width-1;
        slope = compute_slope(ScaleArray,Singular_MSVD, 2, width,i, i,nscales, p );
        while ((i<iMax) && (slope>alpha1))
        { i++;
            i2=i-width+1;
            slope = compute_slope(ScaleArray,Singular_MSVD, 2, width,i2, i,nscales, p );
            
        }
        
        ///find the upper bound
        if (slope<=alpha2)
        {
            l=i;
            while ((l<jMax-1)&& (slope<=alpha2))
            {
                l++;
                l2=l-width+1;
                slope= compute_slope(ScaleArray,Singular_MSVD, 2, width, l2,l,nscales, p );
            }
            l--;
            
            l2=l-width+1;
            
            if (compute_Mean(ScaleArray, Singular_MSVD, width, l2, l, nscales, p)> 0.2)
                
                isaNoisySigularValue= false ;
            else {
                Goodscales[0]=i-1;
                Goodscales[1]=l;}
            
        }// end if upper bound
        else  isaNoisySigularValue=false;
    }
    
    //unsigned long int DimEst;
    DimEst=p;
    
}

/// Ronak Added:

float compute_slope(unsigned long int *ScaleArray,vector<vector<double>> &Singular_MSVD, int method, unsigned long int width,unsigned long int start ,unsigned long int end,unsigned long int iscales,unsigned long int p)
{
    double Multiplys1_sp[end-start+1];
    double Multiply_power_s1[end-start+1];
    double Multiply_power2;
    double sums1_sp=0;
    double sums1=0;
    double sumsp=0;
    double sum4_power_s1=0;
    vector <double> SSP;
    float slope = -1;
    if(Singular_MSVD.empty())
    {
        cout << "No singular values" << endl;
        return 1000;
    }
    else
    {
        for (unsigned long int scale=0; scale<Singular_MSVD.size(); scale++)
            SSP.push_back(Singular_MSVD[scale][p]);
    }
    
    unsigned long int *ScaleArraypor;
    ScaleArraypor = new unsigned long int[end-start+1];
    
    double *SSP_part; // parts of SSP array from start to end with the size width.
    SSP_part = new double[end-start+1];
    // ScaleArray from [start] to [end];
    unsigned long int g;
    if (end<iscales)
    {
        g=0;
        for (unsigned long int ii=start; ii<=end;ii++)
        {
            if (g<=end-start+1)
            {
                ScaleArraypor[g]=ScaleArray[ii];
                SSP_part[g]=SSP[ii];
                g++;
            }
            
        }
        
    }
    
    if (method == 2 && end<iscales)
    {
		
        for (unsigned long int i=0; i<(end-start+1); i++)
        {
            Multiplys1_sp[i]=(double)(ScaleArraypor[i])*(double)(SSP_part[i]);
            sums1_sp+= Multiplys1_sp[i];
            sums1+=(double)ScaleArraypor[i];
            sumsp+=SSP_part[i];
        }
        
        for (unsigned long int i=0; i<(end-start+1); i++)
        {
            Multiply_power_s1[i]=pow((double)(ScaleArray[i]),2);
            
            sum4_power_s1+= Multiply_power_s1[i];
        }
        
        Multiply_power2= pow(sums1,2);
        slope= (sums1_sp-sums1*sumsp/(end-start+1))/ (sum4_power_s1 - Multiply_power2/ (end-start+1));
    }
    
    return slope;
    
}

float compute_Mean(unsigned long int *ScaleArray,vector<vector<double>> &Singular_MSVD, unsigned long int width,unsigned long int start ,unsigned long int end,unsigned long int iscales,unsigned long int p)
//(unsigned long int *ScaleArray,vector<vector<double>> &Singular_MSVD, unsigned long int width, unsigned long int l, unsigned long p)

{
    double MinusSp_S[end-start+1];
    double MinusS_S[end-start+1];
    double divide[end-start+1];
    double Sum=0;
    //if mean((sp(j-width+1:j)-S_MSVD(j-width+1:j,p+1))./(S_MSVD(j-width+1:j,1)-S_MSVD(j-width+1:j,p+1)))> 0.2
    //isaNoisySingularValue = false;
    vector <double> SSP;
    for (unsigned long int scale=0; scale<Singular_MSVD.size(); scale++)
        SSP.push_back(Singular_MSVD[scale][p]);
    
    
    unsigned long int *ScaleArraypor;
    ScaleArraypor = new unsigned long int[end-start+1];
    
    double *SSP_part; // parts of SSP array from start to end with the size width.
    SSP_part = new double[end-start+1];
    // ScaleArray from [start] to [end];
    unsigned long int g;
    if (end<iscales)
    {
        g=0;
        for (unsigned long int ii=start; ii<=end;ii++)
        {
            if (g<=end-start+1)
            {
                ScaleArraypor[g]=ScaleArray[ii];
                SSP_part[g]=SSP[ii];
                g++;
            }
            
        }
        
    }
    
    if(end<iscales)
    {
        unsigned long int st=start;
        for (unsigned long int i=0; i<(end-start+1); i++)
        {
            MinusSp_S[i] =(double)SSP_part[i]-Singular_MSVD[st][p+1];
            MinusS_S[i] =Singular_MSVD[i][0]-Singular_MSVD[st][p+1];
            if (st<end) st++;
            
        }
        
        
        for  (unsigned long int i=0; i<(end-start+1); i++)
        {
            divide[i]=MinusSp_S[i] /MinusS_S[i] ;
            Sum+=divide[i];
        }
    }
    
    double Mean;
    Mean= Sum/width;
    return (Mean);
    
    
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
    // vector<float> A_hat(Matrix[0].size());
    double *degreesMatrix= new double [nPts];
    
    //   double **PointsS, **PointsU;
    // *PointsS = 0;
    // *PointsU = 0;
    
    //MSVD_allocU(PointsS, PointsU,nPts, bags);
    *PointsSS = new double[nPts * bags];
    
    for (unsigned long int k = 0; k < nPts * bags; k++)
    {*(*PointsSS + k) = 0;}
    
    PointsU= new double* [nPts];
    for (unsigned long i=0; i<nPts; i++)
        PointsU[i]= new double[nPts];
    
    
    for (unsigned long i=0; i<nPts;i++)
        for (unsigned long j=0; j<nPts; j++)
        { PointsU[i][j]=0;
            // cout<<"\n vector"<<PointsV[i][j]<<endl;
        }
    
    
    unsigned long KMeans = 3;
    // AT_one= vector<vector<float>>(, vector<float>(nPtsToCompute, std::numeric_limits<float>::quiet_NaN()));
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
    // double **PointsS, **PointsV;
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
    
    //Vectorsvd= vector<double>(Dims, std::numeric_limits<double>::quiet_NaN()));
    
    unsigned long int  j, p, d; // We are going to need indices!
    unsigned long int *nLocalPts;
    // vector <vector<float>> MatrixErr(0);// creating error estimation matrix
    //MatrixErr = vector<vector<float>>(params->nPts, vector<float>(params->NumberOfScales, std::numeric_limits<float>::quiet_NaN()));
    //   float *MatrixErrA=0;
    // Array for how many nn/radius-neighbors are found, for every pt and radius
    nLocalPts = new unsigned long int[params->nPts * params->NumberOfScales];
    unsigned long int highboundry;
    
    unsigned long int EstimateDimension = params->nDims;
    ANNdist *Radii = 0;
    //unsigned long int Goodscales[2]={0,params->NumberOfScales};
    unsigned long  *Goodscales = new unsigned long[2];
    Goodscales[0]=0;
    Goodscales[1]=params->NumberOfScales;
    
    
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
        kNNs = new unsigned long int[params->NumberOfScales];
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
    //vector<double> Vectorsvd(nPtsToCompute);//[atscale][jth singular]
    
    
    double pEPS = 1e-4 ; // precision in nearest neighbor computation
    
    unsigned long *highboundry2= new unsigned long [nPtsToCompute];
    // Allocate memory for the output of nn/radius searches
    nnIdx   = new ANNidx[maxNNs];
    dists  = new ANNdist[maxNNs];
    PointsForSVD = new double[maxNNs * params->nDims];
    //    double* x_c_onepoint= new double [params->nDims];
    double *PowerX_C,*PowerMultiply;
    double *heightminus=0;
    
    //double* X_C = new double[nLocalPts[k * params->NumberOfScales + highboundry]*params->nDims];
    // double* Multiply= new double [nLocalPts[k * params->NumberOfScales + highboundry]*params->nDims];
    
    //vector <unsigned long int> VectScale;
    
    // Array for how many nn/radius-neighbors are found, for every pt and radius
    nLocalPts = new unsigned long int[nPtsToCompute * params->NumberOfScales];
    
    /*--------------------------------------------------
     nPtsToComputeForX_C is the size of whole datasets used which is not neseccarily
     equal to nPtsToCompute
     //--------------------------------------------*/
    
    
    unsigned long nPtsToComputeForX_C = params->nPts;
    
    
    
    
    
    // vector<vector<double>> CreatingA_matrix;
    // CreatingA_matrix= vector<vector<double>>(params->nPts, vector<double>(nPtsToCompute, std::numeric_limits<double>::quiet_NaN()));
    
    /* PointsV= new double* [params->nDims];
     for (unsigned long i=0; i<params->nDims;i++)
     PointsV[i]=new double [params->nDims];
     
     for (unsigned long i=0; i<params->nDims; i++)
     for (unsigned long j=0; j<params->nDims; j++)
     {PointsV[i][j]=0;
     //  cout<<"\n"<<i<<j<<"tamoooooooooooooooooooooomom"<<PointsV[i][j]<<endl;
     
     }*/
    
    double ** CreatingA_matrix= new double* [params->nPts];
    for (unsigned long i=0; i<params->nPts; i++)
        CreatingA_matrix[i]= new double[nPtsToCompute];
    
    
    
    for (unsigned long i=0; i<params->nPts;i++)
        for (unsigned long j=0; j<nPtsToCompute; j++)
        { CreatingA_matrix[i][j]=0;
            // cout<<"\n "<<i<<j<<"tamoooooooooooooooooooooomom"<<CreatingA_matrix[i][j]<<endl;
        }
    
    //loop through the points
    
    for (k = 0; k < nPtsToCompute; k++) {
        // Index of current point
        kCurPt = k + params->startIdx;
        
        cout<<"\n kCurPt"<<kCurPt<<endl;
        // MSVD_alloc_onlyV(params, PointsV);
        
        // Loop through the scales
        for (j = 0; j < params->NumberOfScales; j++)
            
        {
            pthread_mutex_lock(&mutex_1);
            kdTree->annkSearch(dataPts[kCurPt], (int)(kNNs[j]), nnIdx, dists, pEPS);                                // Find nearest neighbors
            pthread_mutex_unlock(&mutex_1);
            nLocalPts[k * params->NumberOfScales + j] = kNNs[j];                                                    // Save number of near neighbors found
            //}
			
            for (p = 0; p < nLocalPts[k * params->NumberOfScales + j]; p++) {                                           // Organize selected points into matrix.
                for (d = 0; d < params->nDims; d++)
                { //cout<<"\n data"<<"k:"<<k<<"j:"<<j<<"d:"<<d<<(dataPts[nnIdx[p]])[d];
                    PointsForSVD[p * params->nDims + d] = (dataPts[nnIdx[p]])[d];
                    //if (j==2){cout<<"\n point for singular values"<<"["<<k<<"]"<<"["<<d<<"]:"<<PointsForSVD[p * params->nDims + d];}
                }
            }
            // Compute the singular values/vectors of the points for this subset of points
            Cov_SVD(PointsForSVD, nLocalPts[k * params->NumberOfScales + j], params->nDims, (*PointsS) + k * params->NumberOfScales * params->nDims + j * params->nDims,(*PointsV)+k*params->NumberOfScales*params->nDims*params->nDims+j*params->nDims*params->nDims);
            for (unsigned long int i=0; i< (params->nDims); i++)
            {
                //cout<<"\n order singular value"<<"["<<k<<"]:"<<"["<<j<<"]:"<<"["<<i<<"]:"<< *((*PointsS) + k * params->NumberOfScales * params->nDims + j * params->nDims+i);
            }
            
            
        }
        //because of Mac test comment
        /* for (unsigned long i = 0; i < params->nDims; i++)
         delete[] PointsV[i];
         delete[] PointsV;*/
        cout<<"\n whyyyyyyyyyy... delete "<<endl;
        //Ronak: here should create a graph to find the lower bound and higher bound for scales to see untill when we increase the scale...like what gualiang did
        EstimateDimFromSpectra(params->nDims,params->NumberOfScales,PointsS,k, kNNs, 5, EstimateDimension, Goodscales);
        cout<<"Run shod"<<endl;
        // highboundry2[k]=Goodscales[1];
        // highboundry=2;//highboundry2[k];
        //EstimateDimension=2;
        
        
        //for testing in MAC
        highboundry=Goodscales[1];
        PointsForMean=new double[kNNs[highboundry] * params->nDims];
        cout<<endl<<endl<<endl<<endl<<endl<<endl;
        cout<<"highoundry= "<<highboundry<<endl;
        cout<<"EstimateDimension= "<<EstimateDimension<<endl;
        cout<<"========================================="<<endl;
        
        cout<<"kNNs[highboundry]"<<kNNs[highboundry];
        
        //    pthread_mutex_lock(&mutex_1);
        // kdTree->annkSearch(dataPts[kCurPt], (int)(kNNs[highboundry]), nnIdx, dists, pEPS);                                // Find nearest neighbors
        //   pthread_mutex_unlock(&mutex_1);
        // nLocalPts[k * params->NumberOfScales + highboundry]=kNNs[highboundry];
        // number of points in high boundry for kth point
        //}
        // testing for MAC
        for (p = 0; p < nLocalPts[k * params->NumberOfScales + highboundry]; p++) {
            for (d = 0; d < params->nDims; d++)
            {
                PointsForMean[p * params->nDims + d] = (dataPts[nnIdx[p]])[d];
                
            }
        }
        
        /*testing correctness
         for (p = 0; p < 2; p++) {
         for (d = 0; d < 2; d++)
         cout<< "\n is correct untill here"<<PointsForMean[p * params->nDims + d]<<endl;}*/
        double** PointsVV;
        PointsVV= new double* [params->nDims];
        for (unsigned long i=0; i<params->nDims; i++)
            PointsVV[i]= new double[params->nDims];
        
        for (unsigned long i=0; i<params->nDims;i++)
            for (unsigned long j=0; j<params->nDims; j++)
                PointsVV[i][j]=*((*PointsV)+k*params->NumberOfScales*params->nDims*params->nDims+highboundry*params->nDims*params->nDims+i*params->nDims+j);
        
        for (unsigned long i=0; i<params->nDims;i++){
            for (unsigned long j=0; j<params->nDims; j++){
                //if(PointsVV[i][j])
                //cout<<"PointsVV[i][j]"<<'\t'<<PointsVV[i][j]<<endl;
            }
        }
        
        
        
        
        //Sum second to last singular values
        //test for MAC
        double SigmaDenominator=0;
        double sumd;
        for (unsigned long int i=1; i< (params->nDims); i++)
        {
            sumd=*((*PointsS) + k * params->NumberOfScales * params->nDims + highboundry * params->nDims+i);
            
            SigmaDenominator+=pow(sumd, 2);}
        double Denominator=2*SigmaDenominator;
        
        
        //test for mac
        
        /*   MSVD_allocV(PointsV, params->nDims, nLocalPts[k * params->NumberOfScales + highboundry], nPtsToCompute);
         
         
         Cov_SVD_withV(PointsForMean, nLocalPts[k * params->NumberOfScales + highboundry], params->nDims,(*PointsS) + k * params->NumberOfScales * params->nDims + highboundry * params->nDims,(*PointsV));*/
        
        /*  for (int i=0;i<5;i++){
         for(int j=0;j<params->nDims;j++){
         cout<<PointsV[i][j]<<'\t';
         }
         cout<<endl;
         }*/
        
        //test for correctness of vectorr
        /* for (unsigned long iV=0; iV<3;iV++)
         for (unsigned d=0; d<4;d++)
         cout<<"\n secondvect:"<<*(*(PointsV)+iV*params->nDims+d);*/
        
        //}// end of k if i use PointsForMean2
        
        
        //------------------
        
        ///// making center and mean for this particular point k and at high boundary scale/////
        
        //test for mac
        double* CntrPt = new double[params->nDims];
        
        // Compute the mean of the points
        unsigned long int xk, yj;
        for (yj = 0; yj < params->nDims; yj++) {
            CntrPt[yj] = 0;
            for (xk = 0; xk < (unsigned long) nLocalPts[k * params->NumberOfScales + highboundry]; xk++)
                CntrPt[yj] += PointsForMean[xk * params->nDims + yj];
            CntrPt[yj] /= nLocalPts[k * params->NumberOfScales + highboundry];
            
        }
        
        //--------------------- calculating Numerator------------
        
        //  MSVD_alloc_2(params,Multiply, highboundry, k , X_C, params->nPts);
        // test for mac
        X_C= new double* [params->nPts];
        for (unsigned long i=0; i<params->nPts; i++)
            X_C[i]= new double[params->nDims];
        
        for (unsigned long i=0; i<params->nPts;i++)
            for (unsigned long j=0; j<params->nDims; j++)
                X_C[i][j]=0;
        
        
        Multiply= new double* [params->nPts];
        for (unsigned long i=0; i<params->nPts; i++)
            Multiply[i]= new double[EstimateDimension];
        
        for (unsigned long i=0; i<params->nPts;i++)
            for (unsigned long j=0; j<EstimateDimension; j++)
                Multiply[i][j]=0;
        
        
        
        //nPtsToComputeForX_C is equal to params-> nPts
        
        /// test for mac
        for (unsigned long iV=0; iV<nPtsToComputeForX_C; iV++)
            
            for (unsigned long d=0; d<params->nDims; d++)
            {
                X_C[iV][d]=dataPts[iV][d]-CntrPt[d];
                
            }
        
        
        
        
        // Matrix multiplication
        // test for mac
        imulti=0;
        Summulti=0;iVmulti=0;
        while(imulti<params->nPts)
        {
            for (unsigned long d=0 ; d<params->nDims; d++)
            {
                Summulti += X_C[imulti][d]* PointsVV[d][iVmulti];
            }
            //before//(*(*(PointsV)+iV*params->nDims+d));
            //before//*(*(PointsV)+iV*params->nDims+d);
            Multiply[imulti][iVmulti]=Summulti;
            Summulti=0;
            if(iVmulti<EstimateDimension)  iVmulti++;
            else {imulti++; iVmulti=0;}
            
        }
        
        // test for correctness
        /*   for (unsigned long l=0; l<params->nPts; l++)
         for (unsigned long l1=0; l1<EstimateDimension; l1++)
         {cout<<Multiply[l][l1]<<"correct untill here"<<'\t';}*/
        
        
        
        
        
        //Numeratoralloc(PowerX_C, PowerMultiply, params->nPts, heightminus);
        // test for mac
        PowerX_C = new double[params->nPts];
        PowerMultiply= new double [params->nPts];
        heightminus= new double [params->nPts];
        
        sum=0;
        sum1=0; 
        for (unsigned long int i=0;i<params->nPts;i++)
        {   for (unsigned long d=0; d<params->nDims; d++)
            sum +=pow(X_C[i][d], 2);
            PowerX_C[i]=sum;
        }
        
        for (unsigned long int i=0;i<params->nPts;i++)
        {
            for (unsigned long int iV=0;iV<EstimateDimension;iV++)
                sum1 +=pow(Multiply[i][iV], 2);
            PowerMultiply[i]=sum1;
        }
        
        
        for (unsigned long int i=0;i<params->nPts;i++)
        {
            heightminus[i]=exp(-abs((PowerX_C[i]- PowerMultiply[i])/Denominator));
            CreatingA_matrix[i][k]=heightminus[i];
            
        }
        
        
        // delete extra matrix and vectors
        // test for mac
        for (unsigned long i = 0; i < params->nDims; i++)
            delete[] Multiply[i];
        delete[] Multiply;
        
        for (unsigned long i = 0; i < params->nPts; i++)
            delete[] X_C[i];
        delete[] X_C;
        for (unsigned long i = 0; i < params->nDims; i++)
            delete[] PointsVV[i];
        delete[] PointsVV;
        
        
        delete[] CntrPt;
        delete[] heightminus;
        delete[] PowerMultiply;
        delete[] PowerX_C;
        
        
        
        
    }// END OF K LOOP
    
    
    
    delete[] kNNs;
    
    delete[] nLocalPts;
    delete[] PointsForSVD;
    
    delete[] nnIdx;
    delete[] dists;
    
    return 0;
}
/*  // Ronak : allocate for creating S1= CDeltas vector of scales//////////////
 unsigned long int *s1=new unsigned long int [params->NumberOfScales];
 for (unsigned long int i=0; i<params->NumberOfScales; i++)
 s1[i]=kNNs[i];*/

///////////////////////////////////////////


/*  void *ComputeMSVD_multi(void *param) {
 // Format input parameter
 param_MSVD_multi *p = (param_MSVD_multi *) param;
 
 //pthread_mutex_lock( &mutex_3 );
 // cout << "\n Thread with startIdx " << (p->paramMSVD).startIdx << " and endIdx " << (p->paramMSVD).endIdx << " called."; flush(cout);
 // pthread_mutex_unlock( &mutex_3 );
 
 // Call the Compute_MSVD routine
 ComputeMSVD(p->dataPts, p->kdTree, &(p->paramMSVD), &(p->PointsS), &(p->PointsV));
 
 return 0;
 }*/




