function [ BaseModale] = Load_bases_MSAV2( nbimage, L, nbmode )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

import pkg_RTI_Fcn.*                         % Import des fonctions RTI, Slopes and Curvatures     

% nbimage = xRTI.Images.nbimage;
% L = xRTI.LP.L ;

if  nbmode >50
BaseModale.nbmode = 50; 
else
BaseModale.nbmode = nbmode; 


% --------     Chargement et construction des bases modales  --------%
BaseModale.basename = 'DEMISPHf';                           % Base demispherique de l'approche Discrete Modal Decomposition (DMD)
[base_mod_locale, normal_elt, Up] = construction_bases_modales_HF(BaseModale.basename,BaseModale.nbmode);

% Construction de la base de projection
mode_interp_decomposition=zeros(nbimage,BaseModale.nbmode);
for i=1:nbimage
    dirL=L(i,:)*1000;
    ind_elt=ondemisphere3D_HF2(dirL,normal_elt);
    elt_target=Up.elt(ind_elt,:);
    x_elt=Up.node(elt_target(1,:),1)';
    y_elt=Up.node(elt_target(1,:),2)';
    z_elt=Up.node(elt_target(1,:),3)';
    
    % Calcul des valeurs (e,n)
    [e,n]=e_n_coordEF(x_elt,y_elt,z_elt,...
        dirL(1,1),dirL(1,2),dirL(1,3));
    
    % Interpolation de la base modale
    if length(Up.elt(1,:))==8
        mode_interp_decomposition(i,:)=interpolation_EF_multi_pts(elt_target(1,:),e,n,base_mod_locale);
    end
end

BaseModale.Pinv_BasemodInterp=(mode_interp_decomposition'*mode_interp_decomposition)\(mode_interp_decomposition');
BaseModale.mode_interp_decomposition=mode_interp_decomposition;

end

