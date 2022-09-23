function io_saveAcquisitionRaw(Acquisition, Path)

    if ~exist(Path, 'dir')
        mkdir(Path);
    end

    for i=1:Acquisition.NbData
        file = fullfile(Path, Acquisition.LP.Names{i});
        if ~exist(file, 'file')
            imwrite(reshape(Acquisition.Data(i,:), Acquisition.DataSize), file);
        end
    end

end

