function [ImagesColorized, ratioNoised, ratioSaturated] = process_colorizeBadValues(Image, limits, colors)

    redChannel = uint8(Image * 255);
    greenChannel = redChannel;
    blueChannel = redChannel;

    noisedMask = Image < limits(1);
    ratioNoised = sum(noisedMask(:)) * 100 / numel(Image);
    redChannel(noisedMask) = colors(1, 1);
    greenChannel(noisedMask) =  colors(1, 2);
    blueChannel(noisedMask) =  colors(1, 3);

    saturatedMask = Image > limits(2);
    ratioSaturated = sum(saturatedMask(:)) * 100 / numel(Image);
    redChannel(saturatedMask) = colors(2, 1);
    greenChannel(saturatedMask) = colors(2, 2);
    blueChannel(saturatedMask) = colors(2, 3);

    ImagesColorized = cat(3, redChannel, greenChannel, blueChannel);
    
end

