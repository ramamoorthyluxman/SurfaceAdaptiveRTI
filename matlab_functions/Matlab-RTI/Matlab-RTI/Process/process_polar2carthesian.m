function [x, y, z] = process_polar2carthesian(theta, phi)
    theta = theta * pi / 180.0;
    phi = phi * pi / 180.0;

    x = cos(phi) .* cos(theta);
    y = cos(phi) .* sin(theta);
    z = sin(phi);
end

