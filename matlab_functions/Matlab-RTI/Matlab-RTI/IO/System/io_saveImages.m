function io_saveImages(Images, ImageSize, Names, path)
    
    for i=1:size(Images, 1)
        io_writeImage(Images(i, :), ImageSize, fullfile(path, Names{i})); 
    end
    
end