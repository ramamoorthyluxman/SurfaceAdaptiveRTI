function F_N = WeigthedLeastSquares(L, RBFn,W)
%weightedLeastSquares Interpolation using weighted least squares
%solution
%   L: the luminance values measured at directions 'dirs' 
%   W: weights for each measurement point to condition the inversion
%   dirs:   [azimuth colatitude] angles in rads for each evaluation point,
%           where colatitude is the polar angle from zenith, size [Ndirsx2] 
%   RBFn: coefficients relative to reflectance basis functions evaluated at
%   directions 'dirs' 
% perform transform in the least squares sense, weighted
%F_N = inv(RBFn'*diag(W)*RBFn)*(RBFn'*diag(W)) * L;
% Last modif 4 mars 2020 yuly C.

%F_N = (RBFn'*diag(W)*RBFn) \ (RBFn'*diag(W)*L);  % weighted 
F_N =pinv(RBFn'*diag(W)*RBFn)*(RBFn'*diag(W)*L);                               
end