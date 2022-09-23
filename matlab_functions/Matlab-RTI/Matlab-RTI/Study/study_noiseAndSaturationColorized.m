function study_noiseAndSaturationColorized(Acquisition1, limits, colors)
    
    directory = 'Dynamic_Limitation';

    if contains(Acquisition1.Path, 'LDR')
        
        for i=1:Acquisition.DataSize(1)
            [I, n, s] = process_colorizeBadValues(Acquisition.Data(i,:), limits, colors);
            [fig, ax, clb] = gui_printData('Image', I, [],  [imageSize 3], 'PixelSize', pixel2mm);
            clb.Visible = 'off';
            io_saveAxis(ax, fullfile(Acquisition.Path, directory, strcat('Colored_saturation_N_', num2str(round(n)), '_S_', num2str(round(s)), '_', Acquisition.LP.Names{i})));
            delete(fig);
        end    
        
    end
    
end

