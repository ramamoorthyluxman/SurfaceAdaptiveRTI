function relighted_img = get_dmd_interpolated_img(theta, phi)

    import pkg_RTI_Fcn.*  

    dmd_mat = load("C:\Users\Ramamoorthy_Luxman\OneDrive - Université de Bourgogne\imvia\work\nblp\SurfaceAdaptiveRTI\matlab_functions\pkg_py\coeffs.mat");
    
    Acquisition = dmd_mat.Acquisition;

    %relighted_img = Reconstruct_DMD_Image(Acquisition.Maps.('DMD').Coefficients, theta,  phi, Acquisition.DataSize(1), Acquisition.DataSize(2), Acquisition.Maps.('DMD').BaseModale);
        
    %imwrite(reshape(relighted_img, Acquisition.DataSize), strcat("E:\acquisitions\LDR_Homogeneous_20220309_155612_Paper\temp\", num2str(theta), num2str(phi), ".png"));
    
    
    relighted_img = zeros(length(theta), prod(Acquisition.DataSize), 'single');
    height = Acquisition.DataSize(2);
    width = Acquisition.DataSize(1);
    for j=1:length(theta)
        relighted_img(j,:) = reshape(Reconstruct_DMD_Image(Acquisition.Maps.('DMD').Coefficients, theta(j),  phi(j), height, width, Acquisition.Maps.('DMD').BaseModale), 1, []);
    end
end

