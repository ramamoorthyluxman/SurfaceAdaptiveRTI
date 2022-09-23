function [fig] = gui_printPixelResponse(Images, index, X, Y, limits)

    fig = figure;
    
    intensity = squeeze(Images(:,index));
    blue = intensity < limits(1);
    red = intensity > limits(2); 

    color = cat(2, red, zeros(length(intensity),1), blue);

    scatter3(X, Y, intensity, 20, color, 'filled');
    xlabel('X');
    ylabel('Y');
    zlabel('Grey Level');
    
end