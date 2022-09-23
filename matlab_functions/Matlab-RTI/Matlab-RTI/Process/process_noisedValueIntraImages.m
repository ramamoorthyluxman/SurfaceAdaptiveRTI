function [DataPercentNoisedIntraImages] = process_noisedValueIntraImages(Images, value)
    DataPercentNoisedIntraImages = (sum(Images < value, 2)) * 100 / size(Images, 2);
end

