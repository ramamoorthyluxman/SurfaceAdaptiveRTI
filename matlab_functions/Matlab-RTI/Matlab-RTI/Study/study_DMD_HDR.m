function study_DMD_HDR(LDR, HDR)

    % HDR.Data data in [0, 1]
    coeff_max = max(HDR.Data(:)) + 1;
    HDR.Data = log(HDR.Data + 1);
    HDR.Data =  HDR.Data / log(coeff_max);
    
    % DMD coefficients 
    MSA = interpolation_dmd_coefficient(HDR.Data, HDR.LP.X, HDR.LP.Y, HDR.LP.Z, 50);
    
    directory = fullfile(HDR.Path, 'DMD', 'Log', LDR.Name);
    io_createDir(directory);
    
    f = waitbar(0.0 / size(HDR.Data, 1), strcat('DMD ... (', num2str(uint8(0.0 * 100 / size(HDR.Data, 1))), '%)' ));
    
    DMD_Mean = zeros(1, size(LDR.Data, 1));
    
    for i=1:size(LDR.Data, 1)
        [theta, phi] = process_carthesian2polar(LDR.LP.X(i), LDR.LP.Y(i), LDR.LP.Z(i));
    
        DMD_Image = interpolation_dmd_reconstruction(MSA, theta, phi, HDR.Data_Size);
        
        exp_time_ms = process_extractExposureTime(LDR.LP.Names{i});
        coeff = exp_time_ms / 500;
        DMD_Image = exp(DMD_Image * log(coeff_max)) - 1;
        DMD_Image = DMD_Image * coeff;
        DMD_Image = process_threshold(DMD_Image, 0, 1);
        
        imwrite(reshape(DMD_Image, HDR.Data_Size(1), HDR.Data_Size(2)), fullfile(directory, LDR.LP.Names{i})); 
    
        Diff = LDR.Data(i, :) - DMD_Image;
    
        [fig, axes, ~] = gui_printData(Diff, HDR.Data_Size, HDR.Pixel_Size, false, 'parula');
        io_saveAxis(axes, fullfile(directory, 'Diff', strcat('Diff', LDR.LP.Names{i}, '.png')) );
        delete(fig);
    
        DMD_Mean(i) = mean(Diff);
        waitbar(i / size(HDR.Data, 1), f, strcat('DMD reconstruction ... (', num2str(uint8(i * 100 / size(HDR.Data, 1))), '%)' ));
    end 
    
    close(f);

    fig = figure;
    histogram(DMD_Mean, 149);
    io_saveFigure(fig, fullfile(directory, 'Diff', 'Distribution_mean_absolute.png'));
    delete(fig);

    fileID = fopen(fullfile(directory,'Diff',  'stat.txt'),'w');
    fprintf(fileID, 'mean of distribution: %f\n',     mean(DMD_Mean));
    fprintf(fileID, 'std of distribution: %f\n',     std(DMD_Mean));
    fclose(fileID);
end