function [Images, m, n] = io_loadImages(Path, Names, Type, Channel)

    if nargin < 3
        Type = 'single';
    end
    
    if nargin < 4
        Channel = 'Gray';
    end
    
    buffer = io_loadImage(Path, Names{1}, Type, Channel);
    [m,n,~]=size(buffer);
    Images = zeros(length(Names), m * n, Type);   

    f = waitbar(0.0 / length(Names), strcat('Load data ... (', num2str(uint8(0.0 * 100 / length(Names))), '%)' ));
    for i=1:length(Names)
        Images(i,:) = reshape(io_loadImage(Path, Names{i}, Type, Channel), 1, []);
        waitbar(i / length(Names), f, strcat('Load data ... (', num2str(uint8(i * 100 / length(Names))), '%)' ));
    end  
    close(f);

end