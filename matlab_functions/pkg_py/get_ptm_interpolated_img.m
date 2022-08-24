function interpolated_img = get_ptm_interpolated_img(lp, ptm_coeffs, h, w)
    import pkg_PTM.* 
    dir = LP_xyz2phitheta(lp);
    interp_pi_new = PTM_terms(dir);
    Image_interp_vect_ptm= interp_pi_new * ptm_coeffs;
    interpolated_img= reshape(Image_interp_vect_ptm, h, w);   
    imwrite(interpolated_img, strcat('C:\Users\Ramamoorthy_Luxman\OneDrive - Université de Bourgogne\imvia\work\nblp\SurfaceAdaptiveRTI\matlab_functions\temp_figures\relighted_img.png'))
end