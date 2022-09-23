function [fig, ax, clb] = gui_printDataSideBySide(Data1, Data2, Data_Size, Pixel_Size, leftText, rightText)
    
    middle = Data_Size(2) / 2;

    buffer1 = reshape(Data1, Data_Size);
    buffer1 = buffer1(:, 1:middle);
    
    buffer2 = reshape(Data2, Data_Size);
    buffer2 = buffer2(:, middle+1:end);
    
    buffer = reshape(cat(2, buffer1, buffer2), 1, []);

    [fig, ax, clb] = gui_printData(buffer, Data_Size, Pixel_Size, false, 'gray');
    clb.Visible = 'off';
    
    line([middle middle], [1 Data_Size(1)], 'LineWidth', 2);
    
    xL=xlim;
    yL=ylim;
    
    if ~isempty(leftText)
        text(0.01*xL(2), 0.99*yL(2), leftText, 'FontSize', 12, 'BackgroundColor', '#DDDDDD', 'HorizontalAlignment', 'left','VerticalAlignment', 'top');
    end
    
    if ~isempty(rightText)
        text(0.99*xL(2),0.99*yL(2), rightText, 'FontSize', 12, 'BackgroundColor', '#DDDDDD', 'HorizontalAlignment', 'right','VerticalAlignment', 'top');
    end
end

