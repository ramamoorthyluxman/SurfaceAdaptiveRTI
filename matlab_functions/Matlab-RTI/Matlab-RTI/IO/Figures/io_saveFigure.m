function io_saveFigure(fig, path)
    
    if nargin < 2
        [file, path] = uigetfile('*.png', 'Choose where to save the figure');
        if isequal(file, 0)
            disp('Selection canceled');
            return;
        end
        path = fullfile(path, file);
    else
        [filepath,~,ext] = fileparts(path) ;
        if ~exist(filepath, 'dir')
            mkdir(filepath)
        end
    end
   
    saveas(fig, path);

end

