function [fitresult, gof] = createFit(X, Y, ag)
%CREATEFIT(X,Y,AG)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : X
%      Y Input : Y
%      Z Output: ag
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 13-Oct-2020 16:01:36


%% Fit: 'untitled fit 1'.
[xData, yData, zData] = prepareSurfaceData( X, Y, ag );

% Set up fittype and options.
ft = 'cubicinterp';

% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, [xData, yData], zData );
colormap 'parula'
legend( h, 'untitled fit 1', 'ag vs. X, Y', 'Location', 'NorthEast' );
% Label axes
xlabel X
ylabel Y
zlabel ag
grid on


