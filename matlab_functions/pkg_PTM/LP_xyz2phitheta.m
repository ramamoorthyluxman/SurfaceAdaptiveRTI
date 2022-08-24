function [dirs] =  LP_xyz2phitheta(LP)
%LP_xyz2phitheta Convert the coordinates of LP=[X,Y,Z] to dirs=[PHI,THETA] 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Gilles Pitard, 26/09/2016
%   gilles.pitard@univ-smb.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % spherical coordinates (Matlab convention)
    [Az,El,~] = cart2sph(LP(:,1),LP(:,2),LP(:,3));
   
    % angles convention of RTI techniques
    PHI = Az;          % phi = azimuth
    THETA = pi/2 - El; % theta = colatitude (the polar angle from zenith)
    dirs = [PHI,THETA];
end