/*
 * File: PTM_terms.c
 *
 * MATLAB Coder version            : 4.1
 * C/C++ source code generated on  : 30-Mar-2020 15:01:57
 */

/* Include Files */
#include "PTM_terms.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *dirs
 *                double P_N[6]
 * Return Type  : void
 */
void PTM_terms(const emxArray_real_T *dirs, double P_N[6])
{
  double a;
  double b_a;
  a = dirs->data[0];
  b_a = dirs->data[1];
  P_N[0] = a * a;
  P_N[1] = b_a * b_a;
  P_N[2] = dirs->data[0] * dirs->data[1];
  P_N[3] = dirs->data[0];
  P_N[4] = dirs->data[1];
  P_N[5] = 1.0;
}

/*
 * File trailer for PTM_terms.c
 *
 * [EOF]
 */
