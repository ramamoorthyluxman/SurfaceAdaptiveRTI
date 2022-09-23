function [Images] = process_threshold(Images, low, high)

    Images(Images < low)  = low;
    Images(Images > high) = high;

end