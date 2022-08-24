 function Q_N =  DMD_getEigenModes(N,dirs, RMB, normal_elt, Up)
%DMD_getEigenModes Get the eigen modes up to a requested number of mode N
%
%   N:  maximum modes
%   dirs:   coordinates [x y z] of light positions 
%   basisType:  'fixed' (i.e. boundary conditions are prescribed and impose
%   no in-plane displacement at every nodes)
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
%dirs=gather(dirs);
ndirs = size(dirs,1);
Q_N = zeros(ndirs,N);

for i=1:ndirs
    dir =dirs(i,:)*1000;% [0.1632, 0.9461, 0.1439]*1000 
    ind_elt= FEM_getElementOntoHemisphere(dir,normal_elt);
    elt_target=Up.elt(ind_elt,:);
    x_elt=Up.node(elt_target(1,:),1)';
    y_elt=Up.node(elt_target(1,:),2)';
    z_elt=Up.node(elt_target(1,:),3)';
    
    % compute (e,n)
    [e,n]=FEM_xyz2en(x_elt,y_elt,z_elt,...
        dir(1,1),dir(1,2),dir(1,3));
    
    %  modal displacements along z are interpolated over a monotonic grid 
    if length(Up.elt(1,:))==8
        Q_N(i,:) = FEM_interp(elt_target(1,:),e,n,RMB);
    end
end
