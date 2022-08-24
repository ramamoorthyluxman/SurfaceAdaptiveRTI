/*
 * File: _coder_PTM_terms_api.h
 *
 * MATLAB Coder version            : 4.1
 * C/C++ source code generated on  : 30-Mar-2020 15:01:57
 */

#ifndef _CODER_PTM_TERMS_API_H
#define _CODER_PTM_TERMS_API_H

/* Include Files */
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include <stddef.h>
#include <stdlib.h>
#include "_coder_PTM_terms_api.h"

/* Type Definitions */
#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T

struct emxArray_real_T
{
  real_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_real_T*/

#ifndef typedef_emxArray_real_T
#define typedef_emxArray_real_T

typedef struct emxArray_real_T emxArray_real_T;

#endif                                 /*typedef_emxArray_real_T*/

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void PTM_terms(emxArray_real_T *dirs, real_T P_N[6]);
extern void PTM_terms_api(const mxArray * const prhs[1], int32_T nlhs, const
  mxArray *plhs[1]);
extern void PTM_terms_atexit(void);
extern void PTM_terms_initialize(void);
extern void PTM_terms_terminate(void);
extern void PTM_terms_xil_terminate(void);

#endif

/*
 * File trailer for _coder_PTM_terms_api.h
 *
 * [EOF]
 */
