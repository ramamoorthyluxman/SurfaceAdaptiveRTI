function gui_printDataOptimized(axes, clb, data)

    valuesMean = nanmean(data);
    valuesSTD = nanstd(data);
    
    ylim(clb, [ min(data(:)) max(data(:)) ])
    caxis(axes, [valuesMean - 4 * valuesSTD valuesMean + 4 * valuesSTD]);
 
end

