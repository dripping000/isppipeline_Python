% ===================================================
% *** FUNCTION linsrgb2xyz
% ***
% *** function [XYZ] = linsrgb2xyz(RGB)
% *** computes XYZ from linear sRGB 
% *** linsRGB is n by 3 and in the range 0-1
% *** XYZ is returned in the range 0-1
% *** see also xyz2linsrgb
% 
% Modified based on the source version from 
% Computational Colour Science using MATLAB 2e
% https://www.mathworks.com/matlabcentral/fileexchange/
% 40640-computational-colour-science-using-matlab-2e
% ===================================================

function [XYZ] = linsrgb2xyz(linsRGB)
if (size(linsRGB,2)~=3)
   disp('RGB must be n by 3'); return;   
end
M = [0.4124 0.3576 0.1805; 0.2126 0.7152 0.0722; 0.0193 0.1192 0.9505];
XYZ = (M*linsRGB')';
XYZ = max(min(XYZ, 1), 0);
end