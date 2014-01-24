/*
 * TestCallToANN.cpp
 *
 *  Created on: Apr 19, 2009
 *      Author: user
 */

//----------------------------------------------------------------------
//		File:			ann_sample.cpp
//		Programmer:		Sunil Arya and David Mount
//		Last modified:	03/04/98 (Release 0.1)
//		Description:	Sample program for ANN
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------

#include <cstdlib>						// C standard library
#include <cstdio>						// C I/O (for sscanf)
#include <cstring>						// string manipulation
#include <fstream>						// file I/O
#include <time.h>
#include "ANN.h"						// ANN declarations

using namespace std;					// make std:: accessible

//----------------------------------------------------------------------
// ann_sample
//
// This is a simple sample program for the ANN library.	 After compiling,
// it can be run as follows.
//
// ann_sample [-d dim] [-max mpts] [-nn k] [-e eps] [-df data] [-qf query]
//
// where
//		dim				is the dimension of the space (default = 2)
//		mpts			maximum number of data points (default = 1000)
//		k				number of nearest neighbors per query (default 1)
//		eps				is the error bound (default = 0.0)
//		data			file containing data points
//		query			file containing query points
//
// Results are sent to the standard output.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Parameters that are set in getArgs()
//----------------------------------------------------------------------
void getArgs(int argc, char **argv);			// get command-line arguments
bool loadPts( FILE *in, unsigned long int *N, unsigned long int *Dim, ANNpointArray *dataPts);


int				k				= 1;			// number of nearest neighbors
double			eps				= 0;			// error bound
bool			verbose         = 1;			// show or not output
FILE*			dataIn			= NULL;			// input for data points



int ParseInputsAndLoadData(int argc, char **argv, unsigned long int *nPts, unsigned long int *nDims, ANNpointArray *dataPts)
{
 	getArgs(argc, argv);						// read command-line arguments

	loadPts(dataIn, nPts, nDims, dataPts);
	fclose(dataIn);

	return 0;
}





void printPt(ostream &out, ANNpoint p, int dim)			// print point
{
	out << "(" << p[0];
	for (int i = 1; i < dim; i++) {
		out << ", " << p[i];
	}
	out << ")\n";
}

bool readHeaderPts( FILE *in, unsigned long int *N, unsigned long int *Dim )
{
	if ( fread(N,8,1 ,in)==1 )					// Read number of points
		if ( fread(Dim,8,1,in)==1 )				// Read dimension of points
			return true;

	return false;
}

bool loadPt( FILE *in, unsigned long int Dim, ANNpoint *dataPt )
{
	return (fread(*dataPt,sizeof(ANNcoord),Dim,in)==Dim);
}

bool loadPts( FILE *in, unsigned long int *N, unsigned long int *Dim, ANNpointArray *dataPts)
{
	if (!readHeaderPts( in, N, Dim ))	return false;

	*dataPts = annAllocPts(*N, *Dim);			// Allocate memory for points

	for (unsigned long int i = 0; i<(*N); i++) {	// Read in the points
		//fread((*dataPts)[i],sizeof(ANNcoord),*Dim,in);
		loadPt(in,*Dim,&((*dataPts)[i]));
	}

	return true;
}

//----------------------------------------------------------------------
//	getArgs - get command line arguments
//----------------------------------------------------------------------

void getArgs(int argc, char **argv)
{
	//static ifstream dataStream;					// data file stream
	//static ifstream queryStream;				// query file stream
	static FILE *dataStream;

	if (argc <= 1) {							// no arguments
		cerr << "Usage:\n\n"
		<< "  FastSpectralGraph [-nn k] [-e eps] [-df data] [-mode mode]"
		   " [-qf query] [-v0]\n\n"
		<< "  where:\n"
		<< "    k        number of nearest neighbors per query (default 1)\n"
		<< "    eps      the error bound (default = 0.0)\n"
		<< "    data     name of file containing data points\n"
		<< "    query    name of file containing query points\n\n"
		<< "    v0       no verbose"
		<< "    mode     how multiple scales are constructed: 0 means by nearest neighbors, 1 means by radii"
		<< " Results are sent to the standard output.\n"
		<< "\n"
		<< " To run this demo use:\n"
		<< "    ann_sample -df data.pts -qf query.pts\n";
		exit(0);
	}
	int i = 1;
	while (i < argc) {							// read arguments
		if (!strcmp(argv[i], "-nn")) {		// -nn option
			k = atoi(argv[++i]);				// get number of near neighbors
		}
		else if (!strcmp(argv[i], "-e")) {		// -e option
			sscanf(argv[++i], "%lf", &eps);		// get error bound
		}
		else if (!strcmp(argv[i], "-df")) {		// -df option
			//dataStream.open(argv[++i], ios::in);// open data file
			dataStream = fopen(argv[++i], "rb");
			if (dataStream==NULL) {
				cerr << "Cannot open data file\n";
				exit(1);
			}
			dataIn = dataStream;				// make this the data stream
		}
		else if (!strcmp(argv[i], "-v0")) {
			verbose = false;
		}
		else if (!strcmp(argv[i], "-mode")) {
		//	sscanf(argv[++i],"%i",&)
		}
		else {									// illegal syntax
			cerr << "Unrecognized option.\n";
			exit(1);
		}
		i++;
	}
	if (dataIn == NULL) {
		cerr << "-df and -qf options must be specified\n";
		exit(1);
	}
}
