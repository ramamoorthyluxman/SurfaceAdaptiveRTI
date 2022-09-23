function io_saveMapsSameDynamic(Acquisition1, Acquisition2, colormap)
    
        maps_names = fieldnames(Acquisition1.Maps);
        maps_names(strcmp(maps_names, 'Saliency')) = [];
        maps_names(strcmp(maps_names, 'Normal')) = [];

        for i=1:size(maps_names, 1)
            if ~isequal(size(Acquisition1.Maps.(maps_names{i})), [0 0]) && ~isequal(size(Acquisition2.Maps.(maps_names{i})), [0 0])
                [fig1, ax1, ~, fig2, ax2, ~] = gui_printDataSameDynamic(Acquisition1.Maps.(maps_names{i}), Acquisition2.Maps.(maps_names{i}), Acquisition1.Data_Size, Acquisition1.Pixel_Size, colormap);
                io_saveAxis(ax1, fullfile(Acquisition1.Path, 'Maps', Acquisition2.Name, maps_names{i}, strcat(maps_names{i}, '_', colormap, '.png')));
                io_saveAxis(ax2, fullfile(Acquisition2.Path, 'Maps', Acquisition1.Name, maps_names{i}, strcat(maps_names{i}, '_', colormap, '.png')));
                close(fig1);
                close(fig2);
            end
        end
end

