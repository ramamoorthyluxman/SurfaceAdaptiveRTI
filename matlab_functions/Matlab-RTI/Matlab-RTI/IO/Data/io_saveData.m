function io_saveData(Mode, Data, Path, varargin)

    [figures, ~, ~] = gui_printData(Mode, Data, varargin{:});
    io_saveFigures(figures, Path);
    
    for i=1:length(figures)
        delete(figures(i));
    end
end

