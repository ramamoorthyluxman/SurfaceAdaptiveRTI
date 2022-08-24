function [modal_basis, normal_elt, Up, dmd_coeffs] = get_dmd_coeffs(images, lps, nb_modes)
    import pkg_DMD.* 
    import pkg_PTM.* 
    import pkg_HSH.* 
    import pkg_fcns.*
    basisname = 'fixed';
    [modal_basis, normal_elt, Up] = DMD_build_RMB(nb_modes,basisname);
    interp_Qi_all = DMD_getEigenModes(nb_modes,lps, modal_basis, normal_elt, Up);
    images = single(images);
    normalised_images=images/255;    
    dmd_coeffs  = LeastSquares(normalised_images, interp_Qi_all); % if weighted add W matrix
end
