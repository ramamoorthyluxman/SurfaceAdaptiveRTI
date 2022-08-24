/*
 * File: LP_xyz2phitheta.c
 *
 * MATLAB Coder version            : 4.1
 * C/C++ source code generated on  : 29-Apr-2020 12:32:32
 */

/* Include Files */
#include <math.h>
#include "rt_defines.h"
#include "rt_nonfinite.h"
#include "LP_xyz2phitheta.h"

/* Function Declarations */
static double rt_atan2d_snf(double u0, double u1);
static double rt_hypotd_snf(double u0, double u1);

/* Function Definitions */

/*
 * Arguments    : double u0
 *                double u1
 * Return Type  : double
 */
static double rt_atan2d_snf(double u0, double u1)
{
  double y;
  int b_u0;
  int b_u1;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = rtNaN;
  } else if (rtIsInf(u0) && rtIsInf(u1)) {
    if (u0 > 0.0) {
      b_u0 = 1;
    } else {
      b_u0 = -1;
    }

    if (u1 > 0.0) {
      b_u1 = 1;
    } else {
      b_u1 = -1;
    }

    y = atan2(b_u0, b_u1);
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = atan2(u0, u1);
  }

  return y;
}

/*
 * Arguments    : double u0
 *                double u1
 * Return Type  : double
 */
static double rt_hypotd_snf(double u0, double u1)
{
  double y;
  double a;
  double b;
  a = fabs(u0);
  b = fabs(u1);
  if (a < b) {
    a /= b;
    y = b * sqrt(a * a + 1.0);
  } else if (a > b) {
    b /= a;
    y = a * sqrt(b * b + 1.0);
  } else if (rtIsNaN(b)) {
    y = b;
  } else {
    y = a * 1.4142135623730951;
  }

  return y;
}

/*
 * LP_xyz2phitheta Convert the coordinates of LP=[X,Y,Z] to dirs=[PHI,THETA]
 *
 * %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 *
 *    Gilles Pitard, 26/09/2016
 *    gilles.pitard@univ-smb.fr
 *
 * %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 * Arguments    : const double LP[150]
 *                double dirs[100]
 * Return Type  : void
 */
void LP_xyz2phitheta(const double LP[150], double dirs[100])
{
  int k;
  double z1[50];

  /*  spherical coordinates (Matlab convention) */
  for (k = 0; k < 50; k++) {
    z1[k] = rt_atan2d_snf(LP[100 + k], rt_hypotd_snf(LP[k], LP[50 + k]));
  }

  /*  angles convention of RTI techniques */
  /*  phi = azimuth */
  /*  theta = colatitude (the polar angle from zenith) */
  for (k = 0; k < 50; k++) {
    dirs[k] = rt_atan2d_snf(LP[50 + k], LP[k]);
    dirs[50 + k] = 1.5707963267948966 - z1[k];
  }
}

/*
 * File trailer for LP_xyz2phitheta.c
 *
 * [EOF]
 */
