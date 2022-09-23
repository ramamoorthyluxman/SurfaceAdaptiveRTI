function [DataPercentSaturatedIntraImages] = process_noisedValueIntraImages(Images, value)
    DataPercentSaturatedIntraImages = (sum(Images > value, 2)) * 100 / size(Images, 2);
end

