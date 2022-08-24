function [e,n]= DMD_xyz2en(x,y,z,x_p,y_p,z_p)
%DMD_xyz2en Calculate the isoparametric coordinates (e,n) of a 8 node
%element in using the global coordinates (x,y,z) of the point test 
%
%   x:  x-coordinates of the 8 nodes of the element
%   y:  y-coordinates of the 8 nodes of the element
%   z:  z-coordinates of the 8 nodes of the element
%   x_p x-coordinate of the test point in the local coordinate system
%   y_p y-coordinate of the test point in the local coordinate system
%   z_p z-coordinate of the test point in the local coordinate system
%
%   [e,n]: isoparametric coordinates over [-1,1]
%
%   previous version fcn: e_n_coordEF.m by H. Favrelière
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Gilles Pitard, 26/09/2016
%   gilles.pitard@univ-smb.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% test of initial conditions
%x_p=gather(x_p);
%y_p=gather(y_p);
%z_p=gather(z_p);
[E,N]=meshgrid(-1:1/3:1,-1:1/3:1);

% associated interpolation function (at the given node)
N1=-1/4*(1-E).*(1-N).*(1+E+N);
N2=-1/4*(1+E).*(1-N).*(1-E+N);
N3=-1/4*(1+E).*(1+N).*(1-E-N);
N4=-1/4*(1-E).*(1+N).*(1+E-N);
N5=1/2*(1-E.^2).*(1-N);
N6=1/2*(1+E).*(1-N.^2);
N7=1/2*(1-E.^2).*(1+N);
N8=1/2*(1-E).*(1-N.^2);

X=[N1;N2;N3;N4;N5;N6;N7;N8].*[x(1)*ones(size(E));x(2)*ones(size(E));...
    x(3)*ones(size(E));x(4)*ones(size(E));x(5)*ones(size(E));...
    x(6)*ones(size(E));x(7)*ones(size(E));x(8)*ones(size(E))];
Y=[N1;N2;N3;N4;N5;N6;N7;N8].*[y(1)*ones(size(E));y(2)*ones(size(E));...
    y(3)*ones(size(E));y(4)*ones(size(E));y(5)*ones(size(E));...
    y(6)*ones(size(E));y(7)*ones(size(E));y(8)*ones(size(E))];
Z=[N1;N2;N3;N4;N5;N6;N7;N8].*[z(1)*ones(size(E));z(2)*ones(size(E));...
    z(3)*ones(size(E));z(4)*ones(size(E));z(5)*ones(size(E));...
    z(6)*ones(size(E));z(7)*ones(size(E));z(8)*ones(size(E))];
X_temp=reshape(X',length(E(1,:)),length(E(:,1)),8);
Y_temp=reshape(Y',length(E(1,:)),length(E(:,1)),8);
Z_temp=reshape(Z',length(E(1,:)),length(E(:,1)),8);

% distance between interpolated points and test point 
ei2_init=[(x_p*ones(size(E'))-sum(X_temp,3)).^2+...
    (y_p*ones(size(E'))-sum(Y_temp,3)).^2+...
    (z_p*ones(size(E'))-sum(Z_temp,3)).^2]';

% indices of the value closest to the test point 
[val_i,ind_i]=min(ei2_init);
[val_min,ind_j]=min(val_i);

X_aff=sum(X_temp,3)';
Y_aff=sum(Y_temp,3)';
Z_aff=sum(Z_temp,3)';
figure(8);axis equal;
plot3([x(1:4) x(1)],[y(1:4) y(1)],[z(1:4) z(1)],'-r');hold on;
plot3(x(5:8),y(5:8),z(5:8),'xb');
plot3(x_p,y_p,z_p,'xr');
plot3(X_aff,Y_aff,Z_aff,'og');
plot3(X_aff(ind_i(ind_j),ind_j),Y_aff(ind_i(ind_j),ind_j),...
    Z_aff(ind_i(ind_j),ind_j),'.k');

savefig(strcat('C:\Users\Ramamoorthy_Luxman\OneDrive - Université de Bourgogne\imvia\work\nblp\SurfaceAdaptiveRTI\matlab_functions\temp_figures\',datestr(now,'HH-MM-SS-FFF')))


% Starting coordinates
e=E(ind_i(ind_j),ind_j);
n=N(ind_i(ind_j),ind_j);

% Optimization problem
k=0;
D=1;
eps=1e-6;

while sum(abs(D))>eps && k<5
    
    N1=-1/4*(1-e)*(1-n)*(1+e+n);
    N2=-1/4*(1+e)*(1-n)*(1-e+n);
    N3=-1/4*(1+e)*(1+n)*(1-e-n);
    N4=-1/4*(1-e)*(1+n)*(1+e-n);
    N5=1/2*(1-e^2)*(1-n);
    N6=1/2*(1+e)*(1-n^2);
    N7=1/2*(1-e^2)*(1+n);
    N8=1/2*(1-e)*(1-n^2);

    dN1_e=1/4*(1-n)*(2*e+n);dN2_e=1/4*(1-n)*(2*e-n);
    dN3_e=1/4*(1+n)*(2*e+n);dN4_e=1/4*(1+n)*(2*e-n);
    dN5_e=-e*(1-n);dN6_e=1/2*(1-n^2);
    dN7_e=-e*(1+n);dN8_e=-1/2*(1-n^2);

    dN1_n=1/4*(1-e)*(2*n+e);dN2_n=1/4*(1+e)*(2*n-e);
    dN3_n=1/4*(1+e)*(2*n+e);dN4_n=1/4*(1-e)*(2*n-e);
    dN5_n=-1/2*(1-e^2);dN6_n=-n*(1+e);
    dN7_n=1/2*(1-e^2);dN8_n=-n*(1-e);

    d2N1_e=1/2*(1-n);d2N2_e=1/2*(1-n);d2N3_e=1/2*(1+n);d2N4_e=1/2*(1+n);
    d2N5_e=-1;d2N6_e=0;d2N7_e=-1;d2N8_e=0;

    d2N1_n=1/2*(1-e);d2N2_n=1/2*(1+e);d2N3_n=1/2*(1+e);d2N4_n=1/2*(1-e);
    d2N5_n=0;d2N6_n=-1;d2N7_n=0;d2N8_n=-1;

    d2N1_en=-1/4*(2*e+2*n-1);d2N2_en=-1/4*(2*e-2*n+1);
    d2N3_en=1/4*(2*e+2*n+1);d2N4_en=1/4*(2*e-2*n+1);
    d2N5_en=e;d2N6_en=-n;d2N7_en=-e;d2N8_en=n;

    dx_e=[dN1_e dN2_e dN3_e dN4_e dN5_e dN6_e dN7_e dN8_e]*x';
    dy_e=[dN1_e dN2_e dN3_e dN4_e dN5_e dN6_e dN7_e dN8_e]*y';
    dz_e=[dN1_e dN2_e dN3_e dN4_e dN5_e dN6_e dN7_e dN8_e]*z';

    dx_n=[dN1_n dN2_n dN3_n dN4_n dN5_n dN6_n dN7_n dN8_n]*x';
    dy_n=[dN1_n dN2_n dN3_n dN4_n dN5_n dN6_n dN7_n dN8_n]*y';
    dz_n=[dN1_n dN2_n dN3_n dN4_n dN5_n dN6_n dN7_n dN8_n]*z';

    d2x_e=[d2N1_e d2N2_e d2N3_e d2N4_e d2N5_e d2N6_e d2N7_e d2N8_e]*x';
    d2y_e=[d2N1_e d2N2_e d2N3_e d2N4_e d2N5_e d2N6_e d2N7_e d2N8_e]*y';
    d2z_e=[d2N1_e d2N2_e d2N3_e d2N4_e d2N5_e d2N6_e d2N7_e d2N8_e]*z';

    d2x_n=[d2N1_n d2N2_n d2N3_n d2N4_n d2N5_n d2N6_n d2N7_n d2N8_n]*x';
    d2y_n=[d2N1_n d2N2_n d2N3_n d2N4_n d2N5_n d2N6_n d2N7_n d2N8_n]*y';
    d2z_n=[d2N1_n d2N2_n d2N3_n d2N4_n d2N5_n d2N6_n d2N7_n d2N8_n]*z';

    d2x_en=[d2N1_en d2N2_en d2N3_en d2N4_en d2N5_en d2N6_en d2N7_en d2N8_en]*x';
    d2y_en=[d2N1_en d2N2_en d2N3_en d2N4_en d2N5_en d2N6_en d2N7_en d2N8_en]*y';
    d2z_en=[d2N1_en d2N2_en d2N3_en d2N4_en d2N5_en d2N6_en d2N7_en d2N8_en]*z';

    X=[N1 N2 N3 N4 N5 N6 N7 N8]*x';
    Y=[N1 N2 N3 N4 N5 N6 N7 N8]*y';
    Z=[N1 N2 N3 N4 N5 N6 N7 N8]*z';

    M11=-2*([d2x_e d2y_e d2z_e]*[x_p-X y_p-Y z_p-Z]'-dx_e^2-dy_e^2-dz_e^2);
    M22=-2*([d2x_n d2y_n d2z_n]*[x_p-X y_p-Y z_p-Z]'-dx_n^2-dy_n^2-dz_n^2);
    M12=-2*([d2x_en d2y_en d2z_en]*[x_p-X y_p-Y z_p-Z]'-dx_e*dx_n-dy_e*dy_n-dz_e*dz_n);
    M21=M12;
    B1=-2*[dx_e dy_e dz_e]*[x_p-X y_p-Y z_p-Z]';
    B2=-2*[dx_n dy_n dz_n]*[x_p-X y_p-Y z_p-Z]';


    M=[M11 M12;M21 M22];
    B=-[B1;B2];

   % D=M\B;
    D=pinv(M)*B ;%yuly

    e=e+D(1);
    n=n+D(2);
    k=k+1   ;
    %chose=sum(abs(D));
end
hold on
% figure(9)
% axis equal;
% plot3([x(1:4) x(1)],[y(1:4) y(1)],[z(1:4) z(1)],'-r');hold on;
% plot3(x(5:8),y(5:8),z(5:8),'xb');
% plot3(x_p,y_p,z_p,'xg');
% plot3(X,Y,Z,'og');

ei=norm([X-x_p;Y-y_p;Z-z_p]);




