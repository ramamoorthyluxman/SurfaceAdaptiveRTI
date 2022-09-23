function [ImagePercentSaturationIntraPixels] = process_badValueIntraPixels(Images, limits)
    ImagePercentSaturationIntraPixels =  process_noisedValueIntraPixels(Images, limits(1)) + process_saturatedValueIntraPixels(Images, limits(2));
end

