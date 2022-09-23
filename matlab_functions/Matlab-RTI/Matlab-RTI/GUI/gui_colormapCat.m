function [color_map] = gui_colormapCat(Data, color_map, threshold, color_below, color_above)
 
    minz = double(min(Data(:)));
    if minz > threshold
      disp('Your range is all above 0, no change');
    else
      maxz = double(max(Data(:)));
      if maxz < threshold
        disp('Your range is all below 0, no change');
      else
        ncol = size(color_map, 1);
        zratio = (threshold - minz) ./ (maxz - minz);
        zpos = max( round(zratio * ncol), 1);
        
        nbColor = size(color_map(1:zpos,:), 1);
        if isstring(color_below)
            color_map(1:zpos,:) = colormap(color_below, nbColor);
        elseif size(color_below) == [2 3]
                color_map(1:zpos,:) = [linspace(color_below(1, 1), color_below(2, 1), nbColor)' ...
                                       linspace(color_below(1, 2), color_below(2, 2), nbColor)' ...
                                       linspace(color_below(1, 3), color_below(2, 3), nbColor)'];
        elseif size(color_below) == [1 3]
            color_map(1:zpos,:) = repmat(color_below, nbColor, 1);
        end
        
        nbColor = size(color_map(zpos+1:end,:), 1);
        if isstring(color_above)
            color_map(zpos+1:end,:) = colormap(color_above, nbColor);
        elseif size(color_above) == [2 3]
                color_map(zpos+1:end,:) = [linspace(color_above(1, 1), color_above(2, 1), nbColor)' ...
                                       linspace(color_above(1, 2), color_above(2, 2), nbColor)' ...
                                       linspace(color_above(1, 3), color_above(2, 3), nbColor)'];
        elseif size(color_above) == [1 3]
            color_map(zpos+1:end,:) = repmat(color_above, nbColor, 1);
        end   
      end
    end
end

