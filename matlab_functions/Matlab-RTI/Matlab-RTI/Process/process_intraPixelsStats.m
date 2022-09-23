function [buffer, meanRatio, stdRatio, minRatio, maxRatio] = process_intraPixelsStats(Images)
    
    buffer = zeros(size(Images, 2), 1);
    parfor i = 1:size(Images, 2)
        tmp = Images(:, i);
        buffer(i) = max(tmp(tmp>0)) / min(tmp(tmp>0));
    end

    meanRatio = mean(buffer);
    stdRatio = std(buffer);
    minRatio = min(buffer);
    maxRatio = max(buffer);
    
end
