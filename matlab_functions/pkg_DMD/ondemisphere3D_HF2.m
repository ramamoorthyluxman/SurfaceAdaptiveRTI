function ind_facet=ondemisphere3D_HF2(pts_test,normal_elt)
% Normalisation à 1 du vecteur
temp_pts_test=pts_test/norm(pts_test);

% Calcul du produit scalaire temp_pts_test*normal_elt
PS=temp_pts_test(1)*normal_elt(:,1)+temp_pts_test(2)*normal_elt(:,2)+temp_pts_test(3)*normal_elt(:,3);

[val,ind_facet]=max(PS);


