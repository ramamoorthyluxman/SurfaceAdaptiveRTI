function relighted_img = get_interpolated_img(lp, modal_basis,  nb_modes, normal_elt, Up, dmd_coeffs, im_h, im_w)
    import pkg_DMD.* 
    import pkg_PTM.* 
    import pkg_HSH.* 
    import pkg_fcns.*
    basisname = 'fixed';
    
    interp_Qi = DMD_getEigenModes(nb_modes, lp, modal_basis(:,1:nb_modes), normal_elt, data.Up);            
    image_interp_vect_DMD = interp_Qi * dmd_coeffs;
    relighted_img =  reshape(image_interp_vect_DMD,im_h,im_w);                
    figure, imshow(relighted_img), title('DMD image'); % show interpolated image

    map= relighted_img - image_interp_mat_DMD2;
    figure, imagesc(map);
end