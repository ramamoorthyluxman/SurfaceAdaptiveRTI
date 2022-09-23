function [exp_time_ms] = process_extractExposureTime(filename)

    [~, name, ~] = fileparts(filename);
    c = strsplit(name, '_');

     j=1;
     while ~contains(c{j}, 'ExposureTime')
         j = j+1;
     end

    exp_time_ms = str2double(c{j+1}) / 1000;
    
end

