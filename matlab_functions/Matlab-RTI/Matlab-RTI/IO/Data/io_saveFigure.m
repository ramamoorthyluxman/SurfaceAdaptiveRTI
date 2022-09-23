function io_saveFigure(Fig, Path)

    if isempty(Path)
        [file, path] = uigetfile('*.png', 'Choose where to save the figure');
        if isequal(file, 0)
            disp('Selection canceled');
            return;
        end
        Path = fullfile(path, file);
    else
        [filepath,~,~] = fileparts(Path) ;
        if ~exist(filepath, 'dir')
            mkdir(filepath)
        end
    end
    
    exportgraphics(Fig, Path, 'BackgroundColor', 'none', 'Resolution', 600)

end

