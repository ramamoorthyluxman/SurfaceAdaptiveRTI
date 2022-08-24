function interpModalBasis = FEM_interp(elt,e,n,modalBasis)
%FEM_interp to return interpolated values at the sample points of every
%modes, relying on a barycentric Interpolation
%
%   elt: indice of element
%   e: 
%   n: 
%   modalBasis: ReflectanceModalBasis
%
%   previous version fcn: interpolation_EF_multi_pts.m by Hugues Favrelière
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Gilles Pitard, 26/09/2016
%   gilles.pitard@univ-smb.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    N7     N6      N5 
%    *------*-------*
%    |              |
%    |              |
% N8 *    8 nodes   *  N4
%    |              |
%    |              |
%    *-------*------*
%    N1      N2     N3


% Interpolation functions
N1=-1/4*(1-e)*(1-n)*(1+e+n); 
N2=-1/4*(1+e)*(1-n)*(1-e+n);
N3=-1/4*(1+e)*(1+n)*(1-e-n);
N4=-1/4*(1-e)*(1+n)*(1+e-n);
N5=1/2*(1-e^2)*(1-n);
N6=1/2*(1+e)*(1-n^2);
N7=1/2*(1-e^2)*(1+n);
N8=1/2*(1-e)*(1-n^2);

% Barycentric Interpolation
interpModalBasis=modalBasis(elt(1,1),:)*N1+modalBasis(elt(1,2),:)*N2...
        +modalBasis(elt(1,3),:)*N3+modalBasis(elt(1,4),:)*N4...
        +modalBasis(elt(1,5),:)*N5+modalBasis(elt(1,6),:)*N6...
        +modalBasis(elt(1,7),:)*N7+modalBasis(elt(1,8),:)*N8;
    
    