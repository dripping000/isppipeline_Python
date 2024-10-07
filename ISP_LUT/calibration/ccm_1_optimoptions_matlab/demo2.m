% This demo shows the color correction pipeline with white point preserved.
% 
% Copyright
% Qiu Jueqin - Feb, 2019

clear; close all; clc;

%% data preparation


%% color correction with white point preserved

% % 指定文件名
% filename = 'src_matlab.txt';
% % 保存为txt文件，默认使用逗号分隔
% dlmwrite(filename, RGB_src, 'delimiter', ' ', 'precision', '%.6f', 'newline', 'pc');
% 
% % 指定文件名
% filename = 'target_matlab.txt';
% % 保存为txt文件，默认使用逗号分隔
% dlmwrite(filename, RGB_target, 'delimiter', ' ', 'precision', '%.6f', 'newline', 'pc');

filename = 'src.txt';
RGB_src = dlmread(filename);
filename = 'target.txt';
RGB_target = dlmread(filename);


% training
model = 'linear3x3';
white_point = [1.00 1.00 1.00];
[matrix, scale, ccm_pred, errs_train] = ccmtrain(RGB_src,...
                                                 RGB_target,...
                                                 'model', model,...
                                                 'targetcolorspace', 'sRGB',...
                                                 'whitepoint', white_point);

% check if [1, 1, 1] has been preserved as [0.9505, 1.0000, 1.0888]
predicted_white_point = ccmapply([1, 1, 1],...
                                 model,...
                                 matrix);
                        
white_point_err = sqrt(sum((predicted_white_point - white_point).^2));
fprintf('residual error between user-specified and predicted white points: %.3e\n', white_point_err);


% visualization
figureFullScreen('color', 'w');
ax1 = subplot(1,2,1);
colors2checker(RGB_src .^ (1/2.2),...
               'layout', [4, 6],...
               'squaresize', 100,...
               'parent', ax1);
title('White balaned camera responses before color correction (gamma = 1/2.2)');

ax2 = subplot(1,2,2);
colors2checker((ccm_pred),...
               'layout', [4, 6],...
               'squaresize', 100,...
               'parent', ax2);
title('Camera responses (sRGB) after color correction');
                        