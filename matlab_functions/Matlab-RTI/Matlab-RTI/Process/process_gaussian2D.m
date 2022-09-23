function [g] = process_gaussian2D(Size, STD)
    
    Size   = (Size-1)/2;

    [x,y] = meshgrid([-Size:Size]/Size,[-Size:Size]/Size);
    g = exp(-(x.*x + y.*y)/(2*STD*STD));

end

