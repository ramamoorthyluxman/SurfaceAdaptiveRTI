function [Image] = interpolation_RBFReconstruction(coeffs, theta, phi)

    L = cat(2, theta, phi);
    size(L')
    Image = rbfinterp(L', coeffs);
    
end

