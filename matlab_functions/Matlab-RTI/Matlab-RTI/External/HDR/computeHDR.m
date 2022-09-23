%function [HDR] = computeHDR(Images, ExposureTimes)
%clear all; close all;
%clc

%% Chargement des images et temps d'exposition
%  Images = double(cat(3,    imread('LDR_000_Theta_2.82562_Phi_17.4452_Exp_218.107.png'), ...
%                                              imread('LDR_000_Theta_2.82562_Phi_17.4452_Exp_62500.png'),...
%                                              imread('LDR_000_Theta_2.82562_Phi_17.4452_Exp_125000.png')));
                       
  Images = double(cat(3,    imread('LDR_054_Theta_354.279_Phi_75_Exp_62500.png'), ...
                                              imread('LDR_054_Theta_354.279_Phi_75_Exp_125000.png')));
                                              %imread('LDR_052_Theta_348.119_Phi_63.9433_Exp_125000.png')));

ExposureTimes = [ 62500, ...
                                125000];

ExposureTimesRef =          125000;
ExposureTimesRelative = ExposureTimes ./ ExposureTimesRef;

%% creation des seuils de saturation et accumulateurs
NOI = 10.0;
SAT = 255.0 - NOI;
Thresholds = [NOI SAT] / 255.0;

height = size(Images, 1);
width = size(Images, 2);

HDR = zeros(height,width, 'double');
counts =  zeros(height,width, 'double');

 %% parametres
 
 weight_type ='Robertson';
%           -weight_type:
%               - 'all': weight is set to 1
%               - 'hat': hat function 1-(2x-1)^12
%               - 'poly': 
%               - 'box': weight is set to 1 in [bounds(1), bounds(2)]
%               - 'Deb97': Debevec and Malik 97 weight function
%               - 'Robertson': a Gaussian with shifting and scaling
 
 bMeanWeight = 0;  % reglage par defaut algo de debevec

 merge_type = 'log';
%           -merge_type:
%               - 'linear': it merges different LDR images in the linear
%               domain.
%               - 'log': it merges different LDR images in the natural
%               logarithmic domain.
%               - 'w_time_sq': it merges different LDR images in the
%               linear; the weight is scaled by the square of the exposure
%               time.
 
lin_type = 'linear';
%               -lin_type: the linearization function:
%                      - 'linear': images are already linear
%                      - 'gamma2.2': gamma function 2.2 is used for
%                                    linearization;
%                      - 'sRGB': images are encoded using sRGB
%                      - 'LUT': the lineraziation function is a look-up
%                               table defined stored as an array in the 
%                               lin_fun 

CRF = [0.002156, 0.002209, 0.002262, 0.002317, 0.002374, ...
		0.002432, 0.002491, 0.002552, 0.002614, 0.002678, ...
		0.002743, 0.002810, 0.002878, 0.002948, 0.003020, ...
		0.003094, 0.003169, 0.003246, 0.003326, 0.003407, ...
		0.003490, 0.003575, 0.003662, 0.003751, 0.003842, ...
		0.003936, 0.004032, 0.004130, 0.004231, 0.004334, ...
		0.004440, 0.004548, 0.004659, 0.004772, 0.004888, ...
		0.005007, 0.005129, 0.005254, 0.005383, 0.005514, ...
		0.005648, 0.005786, 0.005927, 0.006071, 0.006219, ...
		0.006371, 0.006526, 0.006685, 0.006848, 0.007015, ...
		0.007186, 0.007361, 0.007540, 0.007724, 0.007912, ...
		0.008105, 0.008302, 0.008505, 0.008712, 0.008924, ...
		0.009142, 0.009364, 0.009593, 0.009826, 0.010066, ...
		0.010311, 0.010562, 0.010820, 0.011083, 0.011354, ...
		0.011630, 0.011914, 0.012204, 0.012501, 0.012806, ...
		0.013118, 0.013438, 0.013765, 0.014101, 0.014444, ...
		0.014796, 0.015157, 0.015526, 0.015904, 0.016292, ...
		0.016689, 0.017096, 0.017512, 0.017939, 0.018376, ...
		0.018824, 0.019283, 0.019753, 0.020234, 0.020727, ...
		0.021232, 0.021750, 0.022280, 0.022823, 0.023379, ...
		0.023948, 0.024532, 0.025130, 0.025742, 0.026369, ...
		0.027012, 0.027670, 0.028345, 0.029035, 0.029743, ...
		0.030468, 0.031210, 0.031971, 0.032750, 0.033548, ...
		0.034365, 0.035203, 0.036061, 0.036939, 0.037840, ...
		0.038762, 0.039706, 0.040674, 0.041665, 0.042680, ...
		0.043720, 0.044786, 0.045877, 0.046995, 0.048140, ...
		0.049313, 0.050515, 0.051746, 0.053007, 0.054299, ...
		0.055622, 0.056977, 0.058366, 0.059788, 0.061245, ...
		0.062738, 0.064266, 0.065832, 0.067437, 0.069080, ...
		0.070763, 0.072488, 0.074254, 0.076064, 0.077917, ...
		0.079816, 0.081761, 0.083753, 0.085794, 0.087885, ...
		0.090027, 0.092221, 0.094468, 0.096770, 0.099128, ...
		0.101544, 0.104018, 0.106553, 0.109150, 0.111809, ...
		0.114534, 0.117325, 0.120184, 0.123113, 0.126113, ...
		0.129186, 0.132334, 0.135559, 0.138862, 0.142246, ...
		0.145713, 0.149264, 0.152901, 0.156627, 0.160444, ...
		0.164353, 0.168359, 0.172461, 0.176664, 0.180969, ...
		0.185379, 0.189896, 0.194524, 0.199264, 0.204120, ...
		0.209094, 0.214189, 0.219409, 0.224756, 0.230233, ...
		0.235843, 0.241590, 0.247477, 0.253508, 0.259686, ...
		0.266014, 0.272496, 0.279137, 0.285939, 0.292907, ...
		0.300045, 0.307356, 0.314846, 0.322519, 0.330378, ...
		0.338429, 0.346676, 0.355124, 0.363778, 0.372643, ...
		0.381723, 0.391025, 0.400554, 0.410315, 0.420314, ...
		0.430556, 0.441049, 0.451796, 0.462806, 0.474084, ...
		0.485637, 0.497471, 0.509594, 0.522012, 0.534733, ...
		0.547763, 0.561112, 0.574785, 0.588792, 0.603140, ...
		0.617838, 0.632893, 0.648316, 0.664115, 0.680298, ...
		0.696876, 0.713858, 0.731254, 0.749074, 0.767328, ...
		0.786026, 0.805181, 0.824802, 0.844901, 0.865490, ...
		0.886581, 0.908186, 0.930317, 0.952988, 0.976211, ...
		1.000000]';

if strcmp(lin_type, 'LUT')
    lin_fun = CRF;
else
    lin_fun=[];
end

 %is the inverse camera function ok? Do we need to recompute it?
if((strcmp(lin_type, 'LUT') == 1) && isempty(lin_fun))
    [lin_fun, ~] = DebevecCRF(single(stack) / scale, stack_exposure);        
end

%this value is added for numerical stability
delta_value = 1.0 / 65536.0;

%% Reconstruction HDR

 for i=1: length(ExposureTimes)
          
    Image = RemoveCRF(Images(:,:,i) / 255.0, lin_type, lin_fun) * 255.0;
 
     %computing the weight function    
     weight  = WeightFunction(Image / 255.0, weight_type, bMeanWeight, Thresholds);

     % remplissage des acumulateurs
     switch merge_type
            case 'linear'
                HDR = HDR + ( weight .* Image) / ExposureTimesRelative(i);
                counts = counts +  weight;
            case 'log'
                HDR = HDR +  weight .* (log(Image + delta_value) - log(ExposureTimesRelative(i)));
                counts = counts +  weight;                
            case 'w_time_sq'
                HDR = HDR + ( weight .* Image) * ExposureTimesRelative(i);
                counts = counts +  weight * ExposureTimesRelative(i) * ExposureTimesRelative(i);
     end
 end 
 
 HDR = HDR ./ counts;
 
 %% Correction HDR
 if(strcmp(merge_type, 'log') == 1)
     HDR = exp(HDR);
 end
 
% gestion des pixels toujours saturés
 saturation = 0.0001;
 
%  % pixels saturés < NOI dans l'image la plus exposée
% Im = stack1(:,:,1);   % image la plus exposée <=> exposee au temps de reférence d'exposition
 %Im = RemoveCRF(Im/255, lin_type, lin_fun) * 255;
 MedianIndex = length(uint8(ExposureTimesRelative/2));
 pixelOverSaturated = (counts <= saturation) & (Images(:, :, MedianIndex) > 255.0 / 2.0);
 pixelUnderSaturated = (counts <= saturation) & (Images(:, :, MedianIndex) < 255.0 / 2.0);

%  if sum(pixelUnderSaturated(:)) > 0
%     HDR(pixelUnderSaturated) = double(Images(pixelUnderSaturated, end)) ./ ExposureTimesRelative(end);
%  end
 
%  % pixels saturés > SAT dans l'image la moins exposee
 %Im = stack1(:,:,1);   % image la moins exposée
 %Im = RemoveCRF(Im/255, lin_type, lin_fun) * 255;
%   if sum(pixelOverSaturated(:)) > 0
%     HDR(pixelOverSaturated) = double(Images(pixelOverSaturated,1) ./ ExposureTimesRelative(1);
%   end
  
%  ExposureTimeToReconstruct = 14992;
%  LDR = LDR_Reconstruction(HDR, ExposureTimeToReconstruct, ExposureTimesRef);
% figure; imshow(LDR); colorbar;
name = strcat(weight_type, {' '}, merge_type);
% title(strcat(weight_type, {' '}, merge_type));

GUI = uifigure;
%im = uiimage(GUI);
UIAxes = uiaxes(GUI);
UIAxes.Position = [0 0 width/4.0 height/4.0];

sld = uislider(GUI, 'ValueChangedFcn', @(sld,event) updateGauge(sld, GUI, UIAxes, HDR, ExposureTimesRef));
sld.Position = [20 40 900 0];
sld.Limits = [150 125000];

btn = uibutton(GUI, 'Text', 'Video', 'ButtonPushedFcn', @(btn,event) movie(sld, GUI, UIAxes, HDR, ExposureTimesRef));
btn.Position = [20 60  btn.Position(3) btn.Position(4)];

% Create ValueChangedFcn callback
function updateGauge(sld, GUI, UIAxes, HDR, ExposureTimesRef)
    %im.ImageSource = ;
    GUI.Name =  num2str(sld.Value);
    imshow(LDR_Reconstruction(HDR, sld.Value, ExposureTimesRef),'Parent', UIAxes);
end

function movie(sld, GUI, UIAxes, HDR, ExposureTimesRef)
    i = sld.Limits(1);
    while i <= sld.Limits(2)
        sld.Value = i;
        GUI.Name =  num2str(sld.Value);
        imshow(LDR_Reconstruction(HDR, sld.Value, ExposureTimesRef),'Parent', UIAxes); 
        drawnow;
        i = i * 1.25;
    end
end
%  ImageToCompare = Images(:,:,2);
% figure; imshow(uint8(ImageToCompare));
