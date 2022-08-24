function ind_elt = FEM_getElementOntoHemisphere(testPoint,normal_elt)
%FEM_getElementOntoHemisphere(pts_test,normal_elt) Get the indice of the
%element at the test point coordinates 
%
%   testPoints: the global coordinates (x,y,z) of the test point
%   normal_elt: the normal vector of the element
%
%   ind_elt: indice of the element
%
%   previous version fcn: ondemisphere3D_HF2.m by Hugues Favreli?re
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Gilles Pitard, 26/09/2016
%   gilles.pitard@univ-smb.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% normalize the vector to 0-1 range
testPoint=testPoint/norm(testPoint);
%testPoint=gather(testPoint);
% compute the dot product of pts_test*normal_elt
dotP = testPoint(1)*normal_elt(:,1)+...
         testPoint(2)*normal_elt(:,2)+...
         testPoint(3)*normal_elt(:,3);
[~,ind_elt] = (max(dotP));