/*
 * File: _coder_LP_xyz2phitheta_mex.c
 *
 * MATLAB Coder version            : 4.1
 * C/C++ source code generated on  : 29-Apr-2020 12:32:32
 */

/* Include Files */
#include "_coder_LP_xyz2phitheta_api.h"
#include "_coder_LP_xyz2phitheta_mex.h"

/* Function Declarations */
static void LP_xyz2phitheta_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T
  nrhs, const mxArray *prhs[1]);

/* Function Definitions */

/*
 * Arguments    : int32_T nlhs
 *                mxArray *plhs[1]
 *                int32_T nrhs
 *                const mxArray *prhs[1]
 * Return Type  : void
 */
static void LP_xyz2phitheta_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T
  nrhs, const mxArray *prhs[1])
{
  const mxArray *outputs[1];
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;

  /* Check for proper number of arguments. */
  if (nrhs != 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 1, 4,
                        15, "LP_xyz2phitheta");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 15,
                        "LP_xyz2phitheta");
  }

  /* Call the function. */
  LP_xyz2phitheta_api(prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, plhs, outputs);
}

/*
 * Arguments    : int32_T nlhs
 *                mxArray * const plhs[]
 *                int32_T nrhs
 *                const mxArray * const prhs[]
 * Return Type  : void
 */
void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(LP_xyz2phitheta_atexit);

  /* Module initialization. */
  LP_xyz2phitheta_initialize();

  /* Dispatch the entry-point. */
  LP_xyz2phitheta_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  LP_xyz2phitheta_terminate();
}

/*
 * Arguments    : void
 * Return Type  : emlrtCTX
 */
emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/*
 * File trailer for _coder_LP_xyz2phitheta_mex.c
 *
 * [EOF]
 */
