close all; clear all;
import pkg_RTI_Fcn.*                         % Import des fonctions RTI, Slopes and Curvatures     
xRTI = Import_acqs_xRTI;  
xRTI=xRTI(1);
N = 3;                          % Ordre maximum d'harmoniques           (HSH)
K_nbmode = 50;                  % Nombre de modes à l'affichage         (DMD)
npara=10; nbmode=50; 
%% Import base modale 
MSA.BaseModale = Load_bases_MSAV2( xRTI.Images.nbimage, xRTI.LP.L, nbmode );
%% Camputing the coefficients of reconstruction
%%%%PTM
B=[ones(length(xRTI.LP.L(:,1)),1) xRTI.LP.L(:,1) xRTI.LP.L(:,2) xRTI.LP.L(:,1).*xRTI.LP.L(:,2) xRTI.LP.L(:,1).^2 xRTI.LP.L(:,2).^2];
PTM.coef =B\xRTI.Images.Data_vect_norm;
clear B
%%%%HSH
[theta,phi,~] = cart2sph(xRTI.LP.L(:,1),xRTI.LP.L(:,2),xRTI.LP.L(:,3));
dirs = [theta phi];
inclinaison = pi/2 - dirs(:,2);
HSH.coef = weightedLeastSquaresHSHT(N,xRTI.Images.Data_vect_norm, [dirs(:,1) inclinaison], 'real');
%%%%DMD
DMD.coeffs =  MSA.BaseModale.Pinv_BasemodInterp*xRTI.Images.Data_vect_norm;
%Angle of reconstruction
R.theta  = 50; R.phi  = 40;
R.theta_rad=R.theta*pi/180;     R.phi_rad= R.phi *pi/180;
%Reconstructed Images 
PTM_Image=Reconstruct_PTM_Image(PTM.coef,R.theta_rad,R.phi_rad,xRTI.Images.height,xRTI.Images.width);
HSH_Image=Reconstruct_HSH_Image(HSH.coef,R.theta_rad,R.phi_rad,xRTI.Images.height,xRTI.Images.width,N);
DMD_Image=Reconstruct_DMD_Image(DMD.coeffs,R.theta_rad,R.phi_rad,xRTI.Images.height,xRTI.Images.width,MSA.BaseModale );

