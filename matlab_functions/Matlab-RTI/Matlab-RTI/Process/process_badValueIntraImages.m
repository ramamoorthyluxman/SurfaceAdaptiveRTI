function [DataPercentSaturationIntraImages] = process_badValueIntraImages(Images, limits)
    DataPercentSaturationIntraImages =  process_noisedValueIntraImages(Images, limits(1)) + process_saturatedValueIntraImages(Images, limits(2));
end
