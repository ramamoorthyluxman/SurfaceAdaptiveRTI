function [theta, phi] = process_carthesian2polar(x, y, z)

    theta = process_rad2deg(atan2(y,x));
    theta(theta < 0) = theta(theta < 0) + 360;
    
    phi = process_rad2deg(asin(z ./ sqrt(power(x, 2) + power(y, 2) + power(z, 2))));

end

