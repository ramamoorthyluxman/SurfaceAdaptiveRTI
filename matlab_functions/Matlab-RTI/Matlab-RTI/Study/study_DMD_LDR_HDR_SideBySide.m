function study_DMD_LDR_HDR_SideBySide(LDR, HDR)

    % HDR.Data data in [0, 1]
    coeff_max = max(HDR.Data(:)) + 1;
    HDR.Data = log(HDR.Data + 1)/ log(coeff_max);
    
    % DMD coefficients 
    MSA_LDR = interpolation_dmd_coefficient(LDR.Data, LDR.LP.X, LDR.LP.Y, LDR.LP.Z, 50);
    MSA_HDR = interpolation_dmd_coefficient(HDR.Data, HDR.LP.X, HDR.LP.Y, HDR.LP.Z, 50);
    
    directory = fullfile(HDR.Path, 'DMD', 'Log', 'SideBySide', LDR.Name);
    io_createDir(directory);
    
    f = waitbar(0.0 / size(HDR.Data, 1), strcat('DMD ... (', num2str(uint8(0.0 * 100 / size(HDR.Data, 1))), '%)' ));
        
    for i=1:size(LDR.Data, 1)
        
        exp_time_ms = process_extractExposureTime(LDR.LP.Names{i});
        coeff = exp_time_ms / 500;
        
        [theta, phi] = process_carthesian2polar(LDR.LP.X(i), LDR.LP.Y(i), LDR.LP.Z(i));
        DMD_LDR_Image = interpolation_dmd_reconstruction(MSA_LDR, theta, phi, LDR.Data_Size);

        [theta, phi] = process_carthesian2polar(HDR.LP.X(i), HDR.LP.Y(i), HDR.LP.Z(i));
        DMD_HDR_Image = interpolation_dmd_reconstruction(MSA_HDR, theta, phi, HDR.Data_Size);        
        DMD_HDR_Image = process_threshold((exp(DMD_HDR_Image * log(coeff_max)) - 1) * coeff, 0, 1);

        [fig, ax, ~] = gui_printDataSideBySide(DMD_LDR_Image, DMD_HDR_Image, LDR.Data_Size, LDR.Pixel_Size, 'LD-RTI', 'HD-RTI');
        io_saveAxis(ax, fullfile(directory, LDR.LP.Names{i}));
        close(fig);
        
        waitbar(i / size(HDR.Data, 1), f, strcat('DMD reconstruction ... (', num2str(uint8(i * 100 / size(HDR.Data, 1))), '%)' ));
    end 
    
    close(f);

end