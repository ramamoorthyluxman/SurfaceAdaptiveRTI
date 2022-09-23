function io_saveAcquisitionWithOverlay(Acquisition, Index, Title)

    directory = fullfile(process_generatePathAcquisition(Acquisition), 'Overlay');
    
    for i=Index
        file = fullfile(directory, Acquisition.LP.Names{i});
        if ~exist(file, 'file')
            [theta, phi] = process_carthesian2spherical(Acquisition.LP.X(i), Acquisition.LP.Y(i), Acquisition.LP.Z(i));

            [fig, ~, clb] = gui_printData('Image', {Acquisition.Data(i, :)}, Acquisition.DataSize, 'ShadowBanner', 'true', 'Modality', Title, 'Zoom', Acquisition.Zoom, 'PixelSize', Acquisition.PixelSize, 'LightAngle', [theta phi], 'Visible', 'off');
            clb.Visible = 'off';
            io_saveFigure(fig, file);
            close(fig);
        end
    end

end

