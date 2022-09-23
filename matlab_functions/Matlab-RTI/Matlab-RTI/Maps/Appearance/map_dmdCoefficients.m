function [Maps] = map_dmdCoefficients(Maps, Data, LP, nbModes)
%FUNCTION_NAME - One line description of what the function or script performs (H1 line)
%Optional file header info (to give more details about the function than in the H1 line)
%Optional file header info (to give more details about the function than in the H1 line)
%Optional file header info (to give more details about the function than in the H1 line)
%
% Syntax:  [output1,output2] = function_name(input1,input2,input3)
%
% Inputs:
%    input1 - Description
%    input2 - Description
%    input3 - Description
%
% Outputs:
%    output1 - Description
%    output2 - Description
%
% Example: 
%    Line 1 of example
%    Line 2 of example
%    Line 3 of example
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2
% Author: Marvin Nurit
% Work address : 64 b rue de Sully, 21000, Dijon
% email: marvin.nurit@u-bourgogne.fr
% September 2021; Last revision: 27-September-2021

    Maps.('DMD').BaseModale = pkg_RTI_Fcn.Load_bases_MSAV2(size(Data, 1), [LP.X LP.Y LP.Z], nbModes );
    Maps.('DMD').Coefficients =  Maps.('DMD').BaseModale.Pinv_BasemodInterp * Data;
    
end

