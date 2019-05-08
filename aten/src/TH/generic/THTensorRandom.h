#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.h"
#else

TH_API void THTensor_(random)(THTensor *self, THGenerator *_generator);
TH_API void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, int64_t min, int64_t max);
TH_API void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, int64_t max);
TH_API void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(bernoulli_Tensor)(THTensor *self, THGenerator *_generator, THTensor *p);
TH_API void THTensor_(uniform)(THTensor *self, THGenerator *_generator, double a, double b);
TH_API void THTensor_(normal)(THTensor *self, THGenerator *_generator, double mean, double stdv);
TH_API void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
TH_API void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
TH_API void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);
TH_API void THTensor_(exponential)(THTensor *self, THGenerator *_generator, double lambda);
TH_API void THTensor_(cauchy)(THTensor *self, THGenerator *_generator, double median, double sigma);
TH_API void THTensor_(logNormal)(THTensor *self, THGenerator *_generator, double mean, double stdv);
TH_API void THTensor_(multinomial)(THLongTensor *self, THGenerator *_generator, THTensor *prob_dist, int n_sample, int with_replacement);
TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THTensor *q, THLongTensor *J, int n_sample);
#endif

#if defined(TH_REAL_IS_BYTE)
TH_API void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
TH_API void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
#endif

#endif
