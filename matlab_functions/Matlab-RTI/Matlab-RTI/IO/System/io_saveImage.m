function io_saveImage(image, imageSize, path)
    
    [filepath,~,ext] = fileparts(path) ;
    if ~exist(filepath, 'dir')
        mkdir(filepath)
    end
    
    if contains(ext, 'png')
        imwrite(reshape(image, imageSize(1), imageSize(2)), path);
    elseif contains(ext, 'hdr')
        hdrwrite(reshape(image, imageSize(1), imageSize(2)), path);
    end
end