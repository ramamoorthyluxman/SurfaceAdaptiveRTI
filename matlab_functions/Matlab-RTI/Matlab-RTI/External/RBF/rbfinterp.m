function [f] = rbfinterp(x, options)
phi       = options.('rbfphi');
rbfconst  = options.('RBFConstant');
nodes     = options.('x');
rbfcoeff  = (options.('rbfcoeff'))';

[dim, n] = size(options.('x'));
[dimPoints, nPoints] = size(x);
if (dim~=dimPoints)
  error(sprintf('x should have the same number of rows as an array used to create RBF interpolation'));
end;

if nPoints == 1
    r = sqrt(sum(power(x * ones(1, n) - options.('x'), 2), 1));

    s = functions(phi);
    if strcmp(s.function, 'rbfphi_gaussian') 
         f = options.('rbfcoeff')(:, n+1) + sum(options.('rbfcoeff')(:, 1:n) .* exp(-0.5 * power(r, 2)/ power(rbfconst, 2) ), 2) + options.('rbfcoeff')(:, n + 2 : n + 3) * x;
    elseif strcmp(s.function, 'rbfphi_multiquadrics') 
         f = options.('rbfcoeff')(:, n+1) + sum(options.('rbfcoeff')(:, 1:n) .*  sqrt(1+ power(r, 2) / power(rbfconst, 2)) , 2) + options.('rbfcoeff')(:, n + 2 : n + 3) * x;
    elseif strcmp(s.function, 'rbfphi_linear') 
         f = options.('rbfcoeff')(:, n+1) + sum(options.('rbfcoeff')(:, 1:n) .*  r , 2) + options.('rbfcoeff')(:, n + 2 : n + 3) * x;
    elseif  strcmp(s.function, 'rbfphi_cubic') 
          f = options.('rbfcoeff')(:, n+1) + sum(options.('rbfcoeff')(:, 1:n) .*  power(r, 3) , 2) + options.('rbfcoeff')(:, n + 2 : n + 3) * x;
    elseif  strcmp(s.function, 'rbfphi_thinplate') 
          f = options.('rbfcoeff')(:, n+1) + sum(options.('rbfcoeff')(:, 1:n) .*  power(r, 2) .* log(r + 1) , 2) + options.('rbfcoeff')(:, n + 2 : n + 3) * x;
    end
else
    f = zeros(1, nPoints);
    r = zeros(1, n);
    for i=1:1:nPoints
        s=0;
        r =  (x(:,i)*ones(1,n)) - nodes;
        r = sqrt(sum(r.*r, 1));
    %     for j=1:n
    %          r(j) =  norm(x(:,i) - nodes(:,j));
    %     end

         s = rbfcoeff(n+1) + sum(rbfcoeff(1:n).*feval(phi, r, rbfconst)');

        for k=1:dim
           s=s+rbfcoeff(k+n+1)*x(k,i);     % linear part
        end
        f(i) = s;
    end;
end

    if (strcmp(options.('Stats'),'on'))
        fprintf('Interpolation at %d points was computed in %e sec\n', length(f), toc);    
    end;

end

