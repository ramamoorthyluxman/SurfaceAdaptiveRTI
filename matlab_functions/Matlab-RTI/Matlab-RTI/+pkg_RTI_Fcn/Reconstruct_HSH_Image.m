function [HSH_image]=Reconstruct_HSH_Image(HSH_coefs,theta_rad ,phi_rad,Images_height,Image_width,N)
import pkg_RTI_Fcn.* 
inclinaisonE = pi/2 -phi_rad;
Y_N = getHSH(N, [theta_rad inclinaisonE], 'real');
F = Y_N * HSH_coefs;
HSH_image = reshape(F,Images_height,Image_width);  
end