function [phi] = process_extractPhi(filename)

    [~, name, ~] = fileparts(filename);
    c = strsplit(name, '_');

     j=1;
     while ~contains(c{j}, 'Phi')
         j = j+1;
     end

    phi = str2double(c{j+1});
    
end