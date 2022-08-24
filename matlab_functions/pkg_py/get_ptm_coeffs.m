function ptm_coeffs = get_ptm_coeffs(lps, images)
    import pkg_PTM.* 
    dirs= LP_xyz2phitheta(lps);
    interp_Pi= PTM_terms(dirs);
    images = single(images);
    normalised_images=images/255;  
    ptm_coeffs= LeastSquares(normalised_images, interp_Pi);
end




