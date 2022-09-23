function gui_printAddSize(axes,imageSize, pixel2mm)

%     convertX=@(x) sprintf('%.2f', x * pixel2mm(1));
%     convertY =@(x) sprintf('%.2f', x * pixel2mm(2));
% 
%     axes.XTickLabel = cellfun(convertX, num2cell(axes.XTick'), 'UniformOutput', false);
%     axes.YTickLabel = cellfun(convertY, num2cell(axes.YTick'), 'UniformOutput', false);
%     axes.XTickLabelRotation = 45;
%     axes.XLabel.String = 'mm';
%     axes.YLabel.String = 'mm';
    
    axis off ;

    p1 = [imageSize(2) / 40, imageSize(1) / 13];
    p2 = [imageSize(2) / 3, imageSize(1) / 13];
    
    dist = sqrt(sum(power(p1 - p2, 2))) * pixel2mm(1);
    
    xL=xlim;
    yL=ylim;
    
    drawline('Position', [p1; p2] , 'Color', 'black', 'StripeColor', 'white', 'LineWidth', 3);
    text( (p2(1) + p1(1)) / 2.0, p1(2) - imageSize(1) / 30, strcat(num2str(dist, '%.2f'), ' mm'), 'HorizontalAlignment', 'center', 'BackgroundColor', '#DDDDDD', 'FontSize', 8)
    
%     ruler = imdistline(gca, [100 2100], [200 200]);
%     
%     mm = getDistance(ruler);
%     setLabelTextFormatter(ruler, strcat(num2str(mm * pixel2mm(1), '%.2f'), ' mm'));
%     setColor(ruler, 'black');
    
end

