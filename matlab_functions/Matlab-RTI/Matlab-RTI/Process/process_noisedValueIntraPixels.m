function ImagePercentNoisedIntraPixels = process_noisedValueIntraPixels(Images, value)
    ImagePercentNoisedIntraPixels = (sum(Images < value, 1) ) * 100 / size(Images, 1);
end


