function [LP] = io_loadLP(path)
	
    if nargin < 1
        [file, path] = uigetfile('*.lp', 'Choose a Light Positions File (.lp)');
        if isequal(file,0)
            disp('Selection canceled');
            return;
        end
    elseif nargin < 2
        file = dir(fullfile(path, '*.lp'));
    end
    
    LP = struct_LP();
    LP.Path = fullfile(path, file.name);
    
    if exist(LP.Path, 'file') ~= 2
		fprintf("%s n'existe pas ou n'est pas un fichier...\n", LP.Path);
		return
    end
    
    file = fopen(LP.Path, 'r');
    T = textscan(file, '%s %f %f %f', 'HeaderLines', 1);

    buffer=T(1);
    LP.Names=buffer{1};
    LP.X=single(cell2mat(T(2)));
    LP.Y=single(cell2mat(T(3)));
    LP.Z=single(cell2mat(T(4)));	
    
     fclose(file);
end