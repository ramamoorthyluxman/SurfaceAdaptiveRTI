function [w_mmPerPixel, h_mmPerPixel] = process_computeSize(t, pixelSizeMin, pixelSizeMax, imageSize)

    w_mmPerPixel = ((1 - t) * pixelSizeMax(1) + t * pixelSizeMin(1)) / imageSize(2);
    h_mmPerPixel = ((1 - t) * pixelSizeMax(2) + t * pixelSizeMin(2)) / imageSize(1);
    
end

