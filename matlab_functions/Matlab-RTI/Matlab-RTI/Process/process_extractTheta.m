function [theta] = process_extractTheta(filename)

    [~, name, ~] = fileparts(filename);
    c = strsplit(name, '_');

     j=1;
     while ~contains(c{j}, 'Theta')
         j = j+1;
     end

    theta = str2double(c{j+1});
    
end