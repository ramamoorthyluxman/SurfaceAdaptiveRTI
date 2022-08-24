function relighted_img = get_dmd_interpolated_img(lp, modal_basis,  nb_modes, normal_elt, Up, dmd_coeffs, im_h, im_w)
    import pkg_DMD.* 
    import pkg_fcns.*
    
    interp_Qi = DMD_getEigenModes(nb_modes, lp, modal_basis(:,1:nb_modes), normal_elt, Up);            
    image_interp_vect_DMD = interp_Qi * dmd_coeffs;
    relighted_img =  reshape(image_interp_vect_DMD,im_h,im_w);                
    figure, imshow(relighted_img), title('DMD image'); % show interpolated image
    imwrite(relighted_img, strcat('C:\Users\Ramamoorthy_Luxman\OneDrive - Université de Bourgogne\imvia\work\nblp\SurfaceAdaptiveRTI\relighted_img.png'))
end