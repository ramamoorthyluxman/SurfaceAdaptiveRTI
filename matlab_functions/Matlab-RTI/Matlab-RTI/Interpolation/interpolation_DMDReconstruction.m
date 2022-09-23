function [Image] = interpolation_DMDReconstruction(DMD, Theta, Phi, DataSize)

    import pkg_RTI_Fcn.* 

    Theta = deg2rad(Theta);
    Phi = deg2rad(Phi);

    Image = zeros(length(Theta), prod(DataSize), 'single');
    height = DataSize(2);
    width = DataSize(1);
    for j=1:length(Theta)
        Image(j,:) = reshape(Reconstruct_DMD_Image(DMD.Coefficients, Theta(j),  Phi(j), height, width, DMD.BaseModale), 1, []);
    end

end

