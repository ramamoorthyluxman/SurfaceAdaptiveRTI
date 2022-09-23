function io_saveLP(names, x, y, z, path, file)
	   
    if nargin < 6
        file = 'acquisition.lp';
    elseif nargin < 5
        [file, path] = uiputfile('*.lp', 'Choose where to save the positions light file');
        if isequal(file,0)
            disp('Selection canceled');
            return;
        end
    elseif nargin < 4
        disp('Number of argument too small');
        return;
    end
    
    path = fullfile(path, file);

    file=fopen(path, 'w');
    
    % Ecriture des données dans le fichier
    fprintf(file, "%d\n", length(x));
    for i=1:lenght(names)
        fprintf(file, "%s %f %f %f\n", names{i}, x, y, z);
    end
    
    fclose(file);
end