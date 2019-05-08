#ifndef THNN_H
#define THNN_H

#include <stdbool.h>
#include <TH/TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)

#define THIndexTensor THLongTensor
#define THIndexTensor_(NAME) THLongTensor_ ## NAME

#define THIntegerTensor THIntTensor
#define THIntegerTensor_(NAME) THIntTensor_ ## NAME

typedef int64_t THIndex_t;
typedef int32_t THInteger_t;
typedef void THNNState;

#include <THNN/generic/THNN.h>
#include <THGenerateFloatTypes.h>

#endif
