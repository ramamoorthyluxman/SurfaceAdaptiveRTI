function Yi = LeanInterp(X, Y, Xi)
X  = X(:);
Xi = Xi(:);
Y  = Y(:);
nY = numel(Y);
[dummy, Bin] = histc(Xi, X);  %#ok<ASGLU>
H            = diff(X);
if Bin(length(Bin)) >= nY
   Bin(length(Bin)) = nY - 1;
end
Ti = Bin + (Xi - X(Bin)) ./ H(Bin);
% Interpolation parameters:
Si = Ti - floor(Ti);
Ti = floor(Ti);
% Shift frames on boundary:
d     = (Ti == nY);
Ti(d) = Ti(d) - 1;
Si(d) = 1;
% Now interpolate:
Yi = Y(Ti) .* (1 - Si) + Y(Ti + 1) .* Si;