function H_N = HSH_getHarmonics(N, dirs, basisType)
%HSH_getHarmonics Get hemispherical harmonics up to order N
%
%   N:  maximum order
%   dirs:   [azimuth colatitude] angles in radians
%           where colatitude is the polar angle from zenith
%           theta = pi/2-elevation
%   basisType:  complex or real hemispherical harmonics
%
%   H_N: hemispherical harmonics evaluated at the specified directions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Gilles Pitard, 26/09/2016
%   gilles.pitard@univ-smb.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Ndirs = size(dirs, 1);
    Nharm = (N+1)^2;

    H_N = zeros(Nharm, Ndirs);
    idx_Y = 0;
    for n=0:N
        
        m = (-n:n)';
        if isequal(basisType, 'complex')
            % vector of unnormalised associated Legendre functions of current order
            Lnm = legendre2(n, 2*cos(dirs(:,2)')-1);
            
            % normalisations
            norm = sqrt((2*n+1)*factorial(n-m)./ (2*pi*factorial(n+m)));
            
            % convert to matrix, for direct matrix multiplication with the rest
            Nnm = norm * ones(1,Ndirs);
            
            % spherical harmonics of current order
            
            Exp = zeros(2*n+1,Ndirs);
            for m = -n:n
                Exp(m+n+1,:) = exp(1i*m*dirs(:,1)); % stack up azimuthal harmonics in a 3D matrix
            end
           
            Ynm =  Nnm .* Lnm .* Exp;
     
        elseif isequal(basisType, 'real')
            % vector of unnormalised associated Legendre functions of current order
            Lnm_real = legendre(n, 2*cos(dirs(:,2)')-1);
            if n~=0
                Lnm_real = [Lnm_real(end:-1:2, :); Lnm_real];
            end
            
            % normalisations
            norm_real = sqrt((2*n+1)*factorial(n-abs(m)) ./ (2*pi*factorial(n+abs(m))) );
            
            % convert to matrix, for direct matrix multiplication with the rest
            Nnm_real = norm_real * ones(1,Ndirs);
            
            CosSin = zeros(2*n+1,Ndirs);
            for m = -n:n
                if m<0
                    CosSin(m+n+1,:) = sqrt(2)*sin(abs(m)*dirs(:,1)'); % stack up azimuthal harmonics in a 3D matrix
                elseif m == 0
                    CosSin(m+n+1,:) = ones(1,size(dirs,1));
                else
                    CosSin(m+n+1,:) = sqrt(2)*cos(abs(m)*dirs(:,1)');
                end
            end
            Ynm = Nnm_real .* Lnm_real .* CosSin;
            
        end
        
        H_N(idx_Y+1:idx_Y+(2*n+1), :) = Ynm;
        idx_Y = idx_Y + 2*n+1;
    end
    
    % reverse order of the harmonics matrix, size Kx(N+1)^2
    H_N = H_N';
    
end
