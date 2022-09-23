function io_saveFigures(Figures, Paths)
        
    for i=1:length(Figures)
        [filepath,~,~] = fileparts(Paths{i}) ;
        if ~exist(filepath, 'dir')
            mkdir(filepath)
        end

        io_saveFigure(Figures(i), Paths{i});
        close(Figures(i));
    end
    
end

