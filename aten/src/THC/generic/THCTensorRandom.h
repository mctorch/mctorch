#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorRandom.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(uniform)(struct THCState *state, THCTensor *self, double a, double b);
THC_API void THCTensor_(normal)(struct THCState *state, THCTensor *self, double mean, double stdv);
THC_API void THCTensor_(normal_means)(struct THCState *state, THCTensor *self, THCTensor *means, double stddev);
THC_API void THCTensor_(normal_stddevs)(struct THCState *state, THCTensor *self, double mean, THCTensor *stddevs);
THC_API void THCTensor_(normal_means_stddevs)(struct THCState *state, THCTensor *self, THCTensor *means, THCTensor *stddevs);
THC_API void THCTensor_(logNormal)(struct THCState *state, THCTensor *self, double mean, double stdv);
THC_API void THCTensor_(exponential)(struct THCState *state, THCTensor *self, double lambda);
THC_API void THCTensor_(cauchy)(struct THCState *state, THCTensor *self, double median, double sigma);
THC_API void THCTensor_(multinomial)(struct THCState *state, THCudaLongTensor *self, THCTensor *prob_dist, int n_sample, int with_replacement);
THC_API void THCTensor_(multinomialAliasSetup)(struct THCState *state, THCTensor *probs, THCudaLongTensor *J, THCTensor *q);
THC_API void THCTensor_(multinomialAliasDraw)(THCState *state, THCudaLongTensor *self, THCTensor *_q, THCudaLongTensor *_J, int n_sample);

#endif

THC_API void THCTensor_(random)(struct THCState *state, THCTensor *self);
THC_API void THCTensor_(clampedRandom)(struct THCState *state, THCTensor *self, int64_t min, int64_t max);
THC_API void THCTensor_(cappedRandom)(struct THCState *state, THCTensor *self, int64_t max);
THC_API void THCTensor_(geometric)(struct THCState *state, THCTensor *self, double p);

#endif
