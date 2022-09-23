function [meanRatio, stdRatio, minRatio, maxRatio] = process_intraImagesStats(Images)
    
    buffer = zeros(size(Images, 1), 1);
    parfor i = 1:size(Images, 1)
        tmp = Images(:, i);
        buffer(i) = max(tmp(tmp>0)) / min(tmp(tmp>0));
    end

    meanRatio = mean(buffer);
    stdRatio = std(buffer);
    minRatio = min(buffer);
    maxRatio = max(buffer);
    
end

