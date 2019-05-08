#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathReduce.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(renorm)(THCState *state, THCTensor* self, THCTensor* src, scalar_t value, int dimension, scalar_t max_norm);
THC_API void THCTensor_(std)(THCState *state, THCTensor *self, THCTensor *src, int dim, int biased, int keepdim);
THC_API void THCTensor_(norm)(THCState *state, THCTensor* self, THCTensor* src, scalar_t value, int dimension, int keepdim);
THC_API void THCTensor_(var)(THCState *state, THCTensor *self, THCTensor *src, int dim, int biased, int keepdim);

THC_API accreal THCTensor_(stdall)(THCState *state, THCTensor *self, int biased);
THC_API accreal THCTensor_(normall)(THCState *state, THCTensor *self, scalar_t value);
THC_API accreal THCTensor_(varall)(THCState *state, THCTensor *self, int biased);

#endif

THC_API void THCTensor_(prod)(THCState *state, THCTensor *self, THCTensor *src, int dim, int keepdim);

THC_API accreal THCTensor_(sumall)(THCState *state, THCTensor *self);
THC_API accreal THCTensor_(meanall)(THCState *state, THCTensor *self);

THC_API void THCTensor_(min)(THCState *state,
                             THCTensor *values,
                             THCudaLongTensor *indices,
                             THCTensor *src, int dim, int keepdim);
THC_API void THCTensor_(max)(THCState *state,
                             THCTensor *values,
                             THCudaLongTensor *indices,
                             THCTensor *src, int dim, int keepdim);

THC_API scalar_t THCTensor_(minall)(THCState *state, THCTensor *self);
THC_API scalar_t THCTensor_(maxall)(THCState *state, THCTensor *self);
THC_API scalar_t THCTensor_(medianall)(THCState *state, THCTensor *self);

THC_API void THCTensor_(median)(THCState *state,
                                THCTensor *values,
                                THCudaLongTensor *indices,
                                THCTensor *src, int dim, int keepdim);

THC_API accreal THCTensor_(dist)(THCState *state, THCTensor *self, THCTensor *src,
                              scalar_t value);

#endif
