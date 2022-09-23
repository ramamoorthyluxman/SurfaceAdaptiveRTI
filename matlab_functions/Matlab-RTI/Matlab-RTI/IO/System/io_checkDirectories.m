function [checked] = io_checkDirectories(Acquisitions)
    
    checked = true;
    for i=1:size(Acquisitions, 1)
        if ~(exist(Acquisitions{i}{1}, 'dir') == 7)
            disp(Acquisitions{i}{1})
            disp('Not existing')
            checked = false;
            return
        end
    end

end

