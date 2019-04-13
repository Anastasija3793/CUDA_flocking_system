// From: https://github.com/NCCA/cuda_workshops/tree/master/shared/include
#ifndef _RANDOM_H
#define _RANDOM_H

/// Fill up a vector on the device with n floats. Memory is arrumed to have been preallocated.
int randFloats(float *&/*devData*/, const size_t /*n*/);

#endif //_RANDOM_H
