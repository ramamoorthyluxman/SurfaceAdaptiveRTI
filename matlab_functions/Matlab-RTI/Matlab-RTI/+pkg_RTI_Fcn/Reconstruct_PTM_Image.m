function[PTM_image]=Reconstruct_PTM_Image(PTM_coefs,theta_rad ,phi_rad,Images_height,Image_width)
  
  [dirp(1,1),dirp(1,2),dirp(1,3)] = sph2cart(theta_rad ,phi_rad,1);
  xlum = dirp(1,1);
  ylum = dirp(1,2);
  PTM.image_interp_vect=[1 xlum ylum xlum*ylum xlum^2 ylum^2]*PTM_coefs;
  PTM_image= reshape(PTM.image_interp_vect,Images_height,Image_width);  
 
  

    
end
