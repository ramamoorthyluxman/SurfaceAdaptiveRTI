/*
 * File: _coder_LP_xyz2phitheta_api.h
 *
 * MATLAB Coder version            : 4.1
 * C/C++ source code generated on  : 29-Apr-2020 12:32:32
 */

#ifndef _CODER_LP_XYZ2PHITHETA_API_H
#define _CODER_LP_XYZ2PHITHETA_API_H

/* Include Files */
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include <stddef.h>
#include <stdlib.h>
#include "_coder_LP_xyz2phitheta_api.h"

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void LP_xyz2phitheta(real_T LP[150], real_T dirs[100]);
extern void LP_xyz2phitheta_api(const mxArray * const prhs[1], int32_T nlhs,
  const mxArray *plhs[1]);
extern void LP_xyz2phitheta_atexit(void);
extern void LP_xyz2phitheta_initialize(void);
extern void LP_xyz2phitheta_terminate(void);
extern void LP_xyz2phitheta_xil_terminate(void);

#endif

/*
 * File trailer for _coder_LP_xyz2phitheta_api.h
 *
 * [EOF]
 */
