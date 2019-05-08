#include <TH/TH.h>
#include <THNN/THNN.h>

#include <TH/THTensor.hpp>
#include <cmath>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#define THNN_CHECK_SHAPE(I1, I2)                        \
  if (I1 != NULL && I2 != NULL && !THTensor_(isSameSizeAs)(I1, I2))        \
    {                                                        \
       THDescBuff s1 = THTensor_(sizeDesc)(I1);                \
       THDescBuff s2 = THTensor_(sizeDesc)(I2);                \
       THError(#I1 " and " #I2 " shapes do not match: "        \
               #I1 " %s, " #I2 " %s", s1.str, s2.str);        \
    }

#define THNN_CHECK_SHAPE_INDICES(I1, I2)             \
  if (I1 != NULL && I2 != NULL && !I1->sizes().equals(I2->sizes())) \
    {             \
      THDescBuff s1 = THTensor_(sizeDesc)(I1);       \
      THDescBuff s2 = THLongTensor_sizeDesc(I2);     \
      THError(#I1 " and " #I2 " shapes do not match: " \
        #I1 " %s, " #I2 " %s", s1.str, s2.str);      \
    }

#define THNN_CHECK_NELEMENT(I1, I2) \
  if (I1 != NULL && I2 != NULL ) {                                        \
    ptrdiff_t n1 = THTensor_(nElement)(I1);                                        \
    ptrdiff_t n2 = THTensor_(nElement)(I2);                                        \
    if (n1 != n2)                                                        \
      {                                                                        \
        THDescBuff s1 = THTensor_(sizeDesc)(I1);                        \
        THDescBuff s2 = THTensor_(sizeDesc)(I2);                        \
        THError(#I1 " and " #I2 " have different number of elements: "        \
                #I1 "%s has %ld elements, while "                        \
                #I2 "%s has %ld elements", s1.str, n1, s2.str, n2);        \
      }                                                                        \
  }

#define THNN_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE)                        \
  if (THTensor_(nDimensionLegacyNoScalars)(T) != DIM ||                                \
      THTensor_sizeLegacyNoScalars(T, DIM_SIZE) != SIZE) {                                \
      THDescBuff s1 = THTensor_(sizeDesc)(T);                                \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"        \
              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_CHECK_DIM_SIZE_INDICES(T, DIM, DIM_SIZE, SIZE)                        \
  if (THIndexTensor_(nDimensionLegacyNoScalars)(T) != DIM ||                                \
      THTensor_sizeLegacyNoScalars(T, DIM_SIZE) != SIZE) {                                \
      THDescBuff s1 = THIndexTensor_(sizeDesc)(T);                                \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"        \
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_ARGCHECK(COND, ARG, T, FORMAT)        \
  if (!(COND)) {                                \
    THDescBuff s1 = THTensor_(sizeDesc)(T);        \
    THArgCheck(COND, ARG, FORMAT, s1.str);        \
  }

#include <THNN/generic/AbsCriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/BCECriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/ClassNLLCriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/Col2Im.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/ELU.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/HardTanh.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/Im2Col.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/GatedLinearUnit.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/LeakyReLU.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/LogSigmoid.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/MSECriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SoftMarginCriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/MultiLabelMarginCriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/MultiMarginCriterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/RReLU.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/Sigmoid.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SmoothL1Criterion.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SoftPlus.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SoftShrink.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/IndexLinear.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/Tanh.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/TemporalRowConvolution.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/TemporalUpSamplingNearest.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/TemporalUpSamplingLinear.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/FeatureLPPooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/unfold.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialConvolutionMM.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialFullDilatedConvolution.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialDilatedConvolution.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialAveragePooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialDilatedMaxPooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialMaxUnpooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialUpSamplingNearest.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialUpSamplingBilinear.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricAveragePooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialUpSamplingBicubic.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricConvolutionMM.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricFullDilatedConvolution.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricDilatedConvolution.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricAdaptiveAveragePooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricDilatedMaxPooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricMaxUnpooling.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricUpSamplingNearest.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/VolumetricUpSamplingTrilinear.c>
#include <TH/THGenerateFloatTypes.h>

#include <THNN/generic/SpatialClassNLLCriterion.c>
#include <TH/THGenerateFloatTypes.h>
