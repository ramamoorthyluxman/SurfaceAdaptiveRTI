function [ Nb_Im] = countJPG(dirName)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if dirName==0
    fileSorted=[];
    disp(' Empty directory ');
else
    D=dir(dirName);
    [nooffilesf,~]=size(D);
    fileX = [];
    j=0;
    for i=3:nooffilesf
        Nom = D(i).name;
        if strcmp(Nom(end-3:end),'.jpg') ~=1
            j=j+1;
        else
            fileX(i-2-j).name = D(i).name;
        end
    end
    
    [~,Nb_Im]=size(fileX);
end

end

