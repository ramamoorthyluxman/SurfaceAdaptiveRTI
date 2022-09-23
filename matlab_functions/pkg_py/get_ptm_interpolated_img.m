function relighted_img = get_ptm_interpolated_img(theta, phi)

    import pkg_RTI_Fcn.*  

    ptm_mat = load("C:\Users\Ramamoorthy_Luxman\OneDrive - Université de Bourgogne\imvia\work\nblp\SurfaceAdaptiveRTI\matlab_functions\pkg_py\ptm_coeffs.mat");
    
    Acquisition = ptm_mat.Acquisition;

    relighted_img = zeros(length(theta), prod(Acquisition.DataSize), 'single');
    height = Acquisition.DataSize(2);
    width = Acquisition.DataSize(1);
    for j=1:length(theta)
        relighted_img(j,:) = interpolation_PTMReconstruction(Acquisition.Maps.('PTM').Coefficients, theta(j), phi(j));
        %relighted_img(j,:) = reshape(Reconstruct_PTM_Image(Acquisition.Maps.('PTM').Coefficients, theta(j),  phi(j), height, width), 1, []);
    end
end

