%clear all, close all, clc

%import pkg_RTI_Fcn.*        % Import des fonctions RTI
import pkg_DMD.* 
import pkg_PTM.* 
import pkg_HSH.* 
import pkg_fcns.*

% -----------     Import des images       -----------%
%choix du dossier de travail (choisir la premiere image)
[filename,path] = uigetfile({'*.jpg';'*.png';'*.bmp'},'PTM3D : Choisir 1er image');
a=imread([path,filename]);

Images.height=length(a(:,1,1)); 

Images.width=length(a(1,:,1));
Images.depth=length(a(1,1,:));
Images.path = path;
Images.d=dir(path);
Images.filename=filename;

folder1 = Images.path;
fileInfo1 = dir(fullfile(folder1, '*.png'));
imagenb = dir([folder1  '*.png']);
nfiles = length(imagenb);    % Number of files found
for ii=1:nfiles
    currentfilename1 = [folder1 fileInfo1(ii).name];
    if isfile(currentfilename1)
        [currentimage] = (imread(currentfilename1));
    end
    [currentimage] =rgb2gray(currentimage);
    % h=size((currentimage ));
    Images.DataV(:,ii) = (currentimage(:));
end

Images.DataV=single(Images.DataV')./255;
%affichage d'une image en niveau de gris pr verifer
figure(1)
imshow(reshape(Images.DataV(140,:), Images.height, Images.width));


%% code for crop images %%
rect = drawrectangle('Position',[[619.779,268.290,1844.342,1670.404]],'StripeColor','r');%[50,50,300,300],'StripeColor','r');
 for g=1:nfiles
    currentfilename1 = [folder1 fileInfo1(g).name];
    currentimage = (imread(currentfilename1));
    New_im= imcrop( currentimage,rect.Position); 
    Im_name= fileInfo1(g).name; % strcat('Image_',num2str(g,'%03.f%'),'.png');
    folder=('C:\Users\Castro_yuly\Pictures\3D_test_surfaces\test_letters_3_crop');
    imwrite(New_im,fullfile(folder,Im_name));    
 end
%%
for i=1: nfiles     
    Image= Images.Data(:,:,:,i); 
    Images.Red = Image(:,:,1);
    %Images.Green = Image(:,:,2);
    %Images.Blue  = Image(:,:,3);
%figure,  imshow(Images.Red)
%figure, imshow(Images.Green) 
%figure, imshow(Images.Blue)
    %currentfilename1 = [folder1 fileInfo1(i).name];
    %currentimage = (imread(currentfilename1));
    %New_im= imcrop( currentimage,rect.Position); 
    Im_name= fileInfo1(i).name; % strcat('Image_',num2str(g,'%03.f%'),'.png');
    folder=('C:\Users\Castro_yuly\Pictures\Amalia\crop_blue');
    imwrite(Images.Red,fullfile(folder,Im_name));    
end

%Rangement niveaux de gris de chaque pixel en colonnes
[Images.height,Images.width,Images.nbimage ]=size(Im)%Images.Data);
%Images.nbimage =nfiles;
nb_pixels= (Images.height*Images.width);
%Images.Data_vect=reshape(Images.Data(:,:,1,:),Images.height*Images.width,Images.nbimage);%for RGB
Images.Data_vect=reshape(Images.Data,Images.height*Images.width,Images.nbimage);
pixel = single(Images.Data_vect');
Images.pixel_normalise=pixel./255;




clear i pixel nbimage;

%% --------     Chargement et Construction de L (matrice <=> direction d'éclairage)fichier .LP--------%
[LPname,LPpath] = uigetfile('*.lp','LP : Selectionner le fichier d acquisition LP');
fid=fopen([LPpath LPname]);
temp_L=textscan(fid,'%s %f %f %f');  fclose(fid);
Images.LP1=[temp_L{2}(2:end) temp_L{3}(2:end) temp_L{4}(2:end)];
[THETA, PHI, ~]= cart2sph (temp_L{2}(2:end), temp_L{3}(2:end), temp_L{4}(2:end));
deg_theta_LP1=round(wrapTo2Pi(THETA)*180/pi);
deg_phi_LP1=round(PHI*180/pi);

% %pic_files=cell2str(temp_L{1}(2:end))
% for ii=2:156
%     pic_file=char(temp_L{1}(ii:ii));
%     currentfilename1 = [LPpath  pic_file];
%     if isfile(currentfilename1)
%         [currentimage] = (imread(currentfilename1, 'png'));
%     end
%  %  [currentimage] =rgb2gray(currentimage);
%     % h=size((currentimage ));
%     Images.Data(:,:,ii) = (currentimage);
% end
% figure, imshow(Images.Data(:,:,151))
% Images.Data=Images.Data(:,:,2:end);

clear temp_L;

T1 = table(deg_theta_LP1, deg_phi_LP1, 'VariableNames', { 'theta', 'phi'} ); %% conversion LP into theta, phi degrees
 
figure, plotLP(Images.LP1);
figure, plot3(Images.LP1(:,1),Images.LP1(:,2),Images.LP1(:,3), '.-')
tri=delaunay(Images.LP1(:,1),Images.LP1(:,2));
plot (Images.LP1(:,1),Images.LP1(:,2), '.')
[r, c]=size(tri);
figure, trisurf(tri,Images.LP1(:,1),Images.LP1(:,2), Images.LP1(:,3));
%axis vis3d
%axis off
l= light('Position', [-50 -15 29])
set(gca,'CameraPosition', [20 -50 7687])
lighting phong
shading interp


%%  ----------------------------- DMD modal coeficients estimation  -------------- ****

basisname = 'fixed';
data.nb_modes=45;
[data.modal_basis, data.normal_elt, data.Up] = DMD_build_RMB(data.nb_modes,basisname);
interp_Qi_all = DMD_getEigenModes(data.nb_modes,Images.LP1, data.modal_basis,data.normal_elt, data.Up);
data.coef_DMD  = LeastSquares(Images.pixel_normalise, interp_Qi_all); % if weighted add W matrix


%%  ------------- Interpolation in a new lighting direction  DMD -------------------------------
cur_im_dirE(1,1,:)=Images.LP1(3,1,:);   
cur_im_dirE(1,2,:)=Images.LP1(3,2,:);
cur_im_dirE(1,3,:)=Images.LP1(3,3,:);  %% or give xyz coordinates
interp_Qi = DMD_getEigenModes(data.nb_modes, cur_im_dirE, data.modal_basis(:,1:data.nb_modes), data.normal_elt, data.Up);            

% image in a new direction of lighting
image_interp_vect_DMD1= interp_Qi *data.coef_DMD;
image_interp_mat_DMD1=  reshape(image_interp_vect_DMD2,Images.height,Images.width);                
figure, imshow(image_interp_mat_DMD1), title('DMD image'); % show interpolated image

map= image_interp_mat_DMD1- image_interp_mat_DMD2;
figure, imagesc(map);


%% ---------- reconstruction pour tous les directions de depart DMD-------------- ****
for i=1:Images.nbimage
    cur_im_dirE(1,1,:)=Images.LP1(i,1,:);   % Known direction
    cur_im_dirE(1,2,:)=Images.LP1(i,2,:);
    cur_im_dirE(1,3,:)=Images.LP1(i,3,:);

    interp_Qi_new = DMD_getEigenModes(data.nb_modes, cur_im_dirE, data.modal_basis(:,1:data.nb_modes), data.normal_elt, data.Up);
    image_interp_vect_DMD_all= interp_Qi_new *data.coef_DMD;
    image_interp_mat_DMD_all (:,:,i)=  reshape(image_interp_vect_DMD_all,Images.height,Images.width);         

    % save interpolated image
    Image_DMD_all=(image_interp_mat_DMD_all(:,:,i)); 
    Im_name= fileInfo1(i).name; % strcat('Image_',num2str(g,'%03.f%'),'.png');
    folder=('C:\Users\Castro_yuly\Pictures\SUMMUM\zett_old\zett_crop\zet_assem\dmd_zet_ass\');
    imwrite(Image_DMD_all,fullfile(folder,Im_name)); 
     
end
%% --------------------PTM reconsruction -------------------------------------------------- %%
% coeficients estimation

dirs= LP_xyz2phitheta(Images.LP1);
Nterms = 6; %number of terms of the polynome
interp_Pi= PTM_terms(dirs);
%coef_PTM= LeastSquares(Images.pixel_normalise, interp_Pi);
coef_PTM= LeastSquares(Images.DataV, interp_Pi);


% interpolated image to a new direction 
for i=1:nfiles
new_dir= dirs(i,:);   % ar any other dir in phi, theta cordinate coord
interp_pi_new = PTM_terms(new_dir);
Image_interp_vect_ptm= interp_pi_new * coef_PTM;
Image_rep= reshape(Image_interp_vect_ptm, Images.height, Images.width,3);
figure, imshow(Image_rep), title('PTM image')
Im_name= fileInfo1(i).name; % strcat('Image_',num2str(g,'%03.f%'),'.png');
folder=('C:\Users\Castro_yuly\Desktop\trinite_rotate\HSH_rendering');
imwrite(Image_rep_hsh2,fullfile(folder,Im_name)); 
end
 
%% ----------------- HSH reconstruction ------------------------------------------------ %%
% hsh coef estimation

%warning('off','all')
dirs= LP_xyz2phitheta(Images.LP1);
hsh_order= 3; 
intrp_hi=HSH_getHarmonics(hsh_order, dirs,'real');
coef_HSH= LeastSquares(Images.DataV, intrp_hi);% w_vals_rep); if weighted add W matrix

% ------------- interpolation HSH to a new direction 
new_dir= dirs(1,:);
intrp_hi_new2= HSH_getHarmonics(hsh_order, new_dir, 'real');
Image_interp_vect_hsh2= intrp_hi_new2* coef_HSH;
Image_rep_hsh2= reshape(Image_interp_vect_hsh2, Images.height, Images.width,3);
figure, imshow(Image_rep_hsh2), title('hsh_ image')

%for paper
Image_interp_vect_hsh2= intrp_hi_new2 * coef_HSH;
Image_rep_hsh2= reshape(Images.DataV(19,:), Images.height, Images.width,3);
figure, imshow(Image_rep_hsh2), title('Input_ image')
%%%

%hsh all dirs

for i=1:nfiles
    new_dir= dirs(i,:);
    intrp_hi_new2= HSH_getHarmonics(hsh_order, new_dir, 'real');
    Image_interp_vect_hsh2= intrp_hi_new2* coef_HSH;
    Image_rep_hsh2= reshape(Image_interp_vect_hsh2, Images.height, Images.width,3);
    %figure, imshow(Image_rep_hsh2), title('hsh_ image')
    %Image_DMD_all=(image_interp_mat_DMD_all(:,:,i)); 
    Im_name= fileInfo1(i).name; % strcat('Image_',num2str(g,'%03.f%'),'.png');
    folder=('C:\Users\Castro_yuly\Desktop\trinite_rotate\HSH_rendering');
    imwrite(Image_rep_hsh2,fullfile(folder,Im_name)); 
end



%normals
%Normals(Imgages.height, Images.width,3)

Im_norm= Normals(Images.height, Images.width,3, nfiles, Images.DataV, Images.LP1);
figure(6), imshow(Im_norm), 

Im_name=strcat('Im_norm_NBLP','.jpg');
folder=('C:\Users\Castro_yuly\Pictures\ram_data');
imwrite(Im_norm,fullfile(folder,Im_name), 'jpg'); 




% normals = zeros(Images.height,Images.width,3);
% 
%             W=ones(nfiles,1);
%          
%             G= (Images.LP1'*diag(W)*Images.LP1)\ (Images.LP1' *diag(W) * Images.pixel_normalise);
%          
%             normG = sqrt(G(1,:).^2 + G(2,:).^2 + G(3,:).^2);
%             G(1,:)=G(1,:)./normG;
%             G(2,:)=G(2,:)./normG;
%             G(3,:)=G(3,:)./normG;
%             normal2(:,:,1) = reshape(G(1,:), Images.height,Images.width);
%             normal2(:,:,2) = reshape(G(2,:), Images.height,Images.width);
%             normal2(:,:,3) = reshape(G(3,:), Images.height,Images.width);
%             
%             N = round(normal2*100)/100;
%             Nrgb = zeros(Images.height,Images.width,3);
%             % Color codes&
%             Nrgb(:, :, 1)=(N(:,:, 1) + 1) / 2;
%             Nrgb(:, :, 2)=(N(:,:, 2) + 1) / 2;
%             Nrgb(:, :, 3)= N(:,:, 3);
%             im_norm= Nrgb; 
            
            
