function io_createDir(path)

    if ~exist(path, 'dir')
        mkdir(path);
    end
    
end

