function ImagePercentSaturatedIntraPixels = process_saturatedValueIntraPixels(Images, value)
    ImagePercentSaturatedIntraPixels = (sum(Images > value, 1) ) * 100 / size(Images, 1);
end

