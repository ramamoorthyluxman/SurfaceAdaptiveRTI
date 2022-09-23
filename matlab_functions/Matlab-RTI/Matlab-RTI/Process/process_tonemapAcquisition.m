function [Acquisition] = process_tonemapAcquisition(Acquisition)

    directory = 'Tonemap';

    for i=1:Acquisition.Nb_Data
        buffer = reshape(Acquisition.Data(i, :), Acquisition.Data_Size(1), Acquisition.Data_Size(2));
        buffer = tonemap(cat(3, buffer, buffer, buffer));
        Acquisition.Data(i, :) = reshape(buffer(:, :, 1), 1, []);
        Acquisition.LP.Names{i} = strrep(Acquisition.LP.Names{i}, 'hdr', 'png');
        [fig, ax, clb] = gui_printData(Acquisition.Data(i, :), Acquisition.Data_Size, Acquisition.Pixel_Size, false, 'gray');
        clb.Visible = 'off';
        io_saveAxis(ax, fullfile(Acquisition.Path, directory, Acquisition.LP.Names{i}));
        close(fig);
    end
    
end

