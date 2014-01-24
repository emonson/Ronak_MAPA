/*
 * ANN_utils.h
 *
 *  Created on: Apr 19, 2009
 *      Author: user
 */

#ifndef ANN_UTILS_H_
#define ANN_UTILS_H_

#include "ANN.h"						// ANN declarations

int ParseInputsAndLoadData(int argc, char **argv, unsigned long int *nPts, unsigned long int *nDims, ANNpointArray *dataPts);


#endif /* ANN_UTILS_H_ */
