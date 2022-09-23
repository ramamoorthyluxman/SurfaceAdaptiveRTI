function [L, ImgFiles] = getLP(path, filename)
%getLP Extrait les positions des ?clairages du fichier LP (*LightPositions .lp)
%
%  path :       R?pertoire des images
%  filename :   Nom du fichier LP
%
%  L :          Matrice ?clairage, positions des ?clairages en coord. cart?siennes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid=fopen([path filename]);
tline = fgetl(fid);
tline = fgetl(fid);

choice = 'Yes';
temp_L=[];  
temp_L2=[]; % images ignor?es 

while ischar(tline) & choice =='Yes' %#ok<STCMP>
    img_from_lp=textscan(tline,'%s %f %f %f');
 
   if tline(1)~='#'
        chemin_img_parcourue = [path char(img_from_lp{1})];
        
        % si image sp?cifi?e dans le LP n'existe pas
        if exist(chemin_img_parcourue, 'file')~=2
            % boite de dialogue
            qstring = ['Image file ' char(img_from_lp{1}) ' doesn''t exist. Do you want to continue?'];
            choice = questdlg(qstring,'File error', 'No');
        else
            temp_L=[temp_L ; img_from_lp];
        end
   end
   temp_L2=[temp_L2; img_from_lp]; 
   tline = fgetl(fid);
end
fclose(fid);

%% Construction de L (matrice ?clairage)
L = [temp_L(:,2) temp_L(:,3) temp_L(:,4)]; 
L2= [temp_L2(:,2) temp_L2(:,3) temp_L2(:,4)];
ImgFiles  = temp_L(:,1);
L=cell2mat(L);
L2=cell2mat(L2);
end