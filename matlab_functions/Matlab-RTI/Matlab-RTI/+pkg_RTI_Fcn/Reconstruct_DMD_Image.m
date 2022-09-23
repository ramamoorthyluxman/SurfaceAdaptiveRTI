function [DMD_image]= Reconstruct_DMD_Image(DMD_coeffs,theta_rad ,phi_rad,Images_height,Image_width,BaseModale)
import pkg_RTI_Fcn.* 
[BaseModale.base_mod_locale, BaseModale.normal_elt, BaseModale.Up] = construction_bases_modales_HF(BaseModale.basename,  BaseModale.nbmode);
% Calcul des valeurs theta, phi en coordonnées cart
[dirE(1,1),dirE(1,2),dirE(1,3)]=sph2cart(theta_rad,phi_rad,1);

ind_elt = ondemisphere3D_HF2(dirE ,BaseModale.normal_elt);
elt_target = BaseModale.Up.elt(ind_elt,:);

x_elt=BaseModale.Up.node(elt_target(1,:),1)';
y_elt=BaseModale.Up.node(elt_target(1,:),2)';
z_elt=BaseModale.Up.node(elt_target(1,:),3)';
[e,n]=e_n_coordEF(x_elt,y_elt,z_elt,dirE(1,1),dirE(1,2),dirE(1,3));

% Interpolation de la base modale
mode_interp= interpolation_EF_multi_pts(elt_target(1,:),e,n,BaseModale.base_mod_locale(:,1 :BaseModale.nbmode));
DMD.image_interp_vect = mode_interp*DMD_coeffs;     
DMD_image= reshape(DMD.image_interp_vect,Images_height,Image_width);  
end