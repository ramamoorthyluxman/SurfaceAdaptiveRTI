function [xRTI] = Import_acqs_xRTI
%% Chargement d'un ensemble d'acquisition %

aDirectory = dir(uigetdir('C:\Users\DELL_RTI_PORTABLE_01\Documents\Programmation\Acqu\'));
j=1;
    for y =3:numel(aDirectory)
       
       xRTI(j).LP.name='acquisition.lp';
       xRTI(j).path=strcat(aDirectory(y).folder,'\',aDirectory(y).name,'\');
       fid=fopen([xRTI(j).path xRTI(j).LP.name]);
       temp_L=textscan(fid,'%s %f %f %f');  fclose(fid);
       xRTI(j).LP.L=[temp_L{2}(2:end) temp_L{3}(2:end) temp_L{4}(2:end)];
       xRTI(j).Images.nbimage = length(xRTI(j).LP.L);

       %% -----------     Import des images       -----------%
        a=imread([xRTI(j).path,temp_L{1,1}{2,1}]);  % temp_L{1,1}{2,1} <=> nom de la premi?re image d aquisition
        xRTI(j).Images.height=length(a(:,1,1));     xRTI(j).Images.width=length(a(1,:,1));
        s=size(a,3);    % gestion de la couleur (si acquisition en couleur)

        %  Importation des images de l'objet("nbimage" images, stockees dans Data(i,j,1:nbimage))
        xRTI(j).Images.Data=uint8(zeros(xRTI(j).Images.height,xRTI(j).Images.width,xRTI(j).Images.nbimage));

        %formatSpec = '%03d\n';
        for i=1:xRTI(j).Images.nbimage
            if s==1 
            xRTI(j).Images.Data(:,:,i)= imread([xRTI(j).path,temp_L{1,1}{i+1,1}])  ;  end
            if s==3
            xRTI(j).Images.Data(:,:,i)= rgb2gray(imread([xRTI(j).path,temp_L{1,1}{i+1,1}]))  ;  end
        end
        % Rangement niveaux de gris de chaque pixel en colonnes
        xRTI(j).Images.Data_vect=reshape(xRTI(j).Images.Data,xRTI(j).Images.height*xRTI(j).Images.width,xRTI(j).Images.nbimage);
        xRTI(j).Images.Data_vect_norm=double(xRTI(j).Images.Data_vect')./255;
    j=j+1;
    end
end