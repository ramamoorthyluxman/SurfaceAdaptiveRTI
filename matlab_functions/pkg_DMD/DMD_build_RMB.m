function [RMB, normal_elt, Up] =  DMD_build_RMB(N,basisName)
%DMD_build_RMB Build the reflectance modal basis (RMB)
%
%   N:  maximum modes
%   basisType:  'fixed' or others.
%
%   RMB: Reflectance Modal Basis
%   normal_elt:
%   Up:
%
%   previous version fcn: construction_bases_modales_HF.m by H. Favrelière
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Gilles Pitard, 26/09/2016
%   gilles.pitard@univ-smb.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(basisName, 'fixed')
    fprintf("using fixed")
    load('DEMISPHf.mat'); %old base
    %load('Base_new.mat'); %%yuly new base 
else % not fixed
    load('DEMISPH.mat');
end

RMB  = zeros(length(defANSYS.def(:,1)),N,'double');
for i=1:N
    RMB(:,i)=defANSYS.def(:,i)/max(abs(defANSYS.def(:,i)));
end

% RMB  = zeros(length(Bases.def(:,1)),N,'double');   %%modificacion base yuly
% for i=1:N
%     RMB(:,i)=Bases.def(:,i)/max(abs(Bases.def(:,i)));
% end


% approximate normal vectors using the barycenter method
P1=(Up.node(Up.elt(:,3),:)+Up.node(Up.elt(:,1),:))./2;
P2=(Up.node(Up.elt(:,4),:)+Up.node(Up.elt(:,2),:))./2;
PC=(P1+P2)./2;
normal_elt = PC./[(PC(:,1).^2+PC(:,2).^2+PC(:,3).^2).^(1/2) ...
    (PC(:,1).^2+PC(:,2).^2+PC(:,3).^2).^(1/2) ...
    (PC(:,1).^2+PC(:,2).^2+PC(:,3).^2).^(1/2)];
Up.node =round(Up.node*1e4)/1e4;
end