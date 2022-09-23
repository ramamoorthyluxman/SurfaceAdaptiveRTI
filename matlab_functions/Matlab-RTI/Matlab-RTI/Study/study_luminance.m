function study_luminance(Acquisition)
    
    directory = 'Luminance';
    for i=1:100000:size(Acquisition.Data, 2)
        io_saveData('Plot', Acquisition.Data(:,i) * 255, fullfile(directory, strcat(num2str(i), '.png')), [], [], Acquisition.LP.X, Acquisition.LP.Y, 'Unit', 'Gray Level')
    end

end

