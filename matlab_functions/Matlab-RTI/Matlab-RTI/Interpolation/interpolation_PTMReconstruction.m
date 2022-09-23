function [Image] = interpolation_PTMReconstruction(coeff, theta, phi)

    %[x, y, ~] = process_spherical2carthesian(theta, phi);
    [x, y, ~] = sph2cart(theta, phi, 1);

    l = cat(2, x .* x, y .* y, x .* y, x, y, ones(size(x)));

    Image = l * coeff;

end