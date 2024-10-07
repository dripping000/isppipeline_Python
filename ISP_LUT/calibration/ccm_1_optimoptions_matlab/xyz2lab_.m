% ===================================================
% *** FUNCTION xyz2lab
% ***
% *** function [lab] = xyz2lab(xyz, obs, xyzw)
% *** computes LAB from XYZ 
% *** xyz is an n by 3 matrix 
% *** e.g. set obs to 'd65_64 for D65 and 1964
% *** set obs to 'user' to use optional argument   
% *** xyzw as the white point
% 
% IMPORTANT NOTE:
% Input xyz must be within range [0, 100] instead of [0, 1] !!!
%
% Modified based on the source version from 
% Computational Colour Science using MATLAB 2e
% https://www.mathworks.com/matlabcentral/fileexchange/
% 40640-computational-colour-science-using-matlab-2e
% ===================================================

function [lab] = xyz2lab_(xyz,obs,xyzw)
 
if nargin < 2
    obs = 'd65_31'; % default obs
end
if (size(xyz,2)~=3)
   disp('xyz must be n by 3'); return;   
end
lab = zeros(size(xyz,1),size(xyz,2));

if strcmp('a_64',obs)
    white=[111.144 100.00 35.200];
elseif strcmp('a_31', obs)
    white=[109.850 100.00 35.585];
elseif strcmp('c_64', obs)
    white=[97.285 100.00 116.145];
elseif strcmp('c_31', obs)
    white=[98.074 100.00 118.232];
elseif strcmp('d50_64', obs)
    white=[96.720 100.00 81.427];
elseif strcmp('d50_31', obs)
    white=[96.422 100.00 82.521];
elseif strcmp('d55_64', obs)
    white=[95.799 100.00 90.926];
elseif strcmp('d55_31', obs)
    white=[95.682 100.00 92.149];
elseif strcmp('d65_64', obs)
    white=[94.811 100.00 107.304];
elseif strcmp('d65_31', obs)
    white=[95.047 100.00 108.883];
elseif strcmp('d75_64', obs)
    white=[94.416 100.00 120.641];
elseif strcmp('d75_31', obs)
    white=[94.072 100.00 122.638];
elseif strcmp('f2_64', obs)
    white=[103.279 100.00 69.027];
elseif strcmp('f2_31', obs)
    white=[99.186 100.00 67.393];
elseif strcmp('f7_64', obs)
    white=[95.792 100.00 107.686];
elseif strcmp('f7_31', obs)
    white=[95.041 100.00 108.747];
elseif strcmp('f11_64', obs)
    white=[103.863 100.00 65.607]; 
elseif strcmp('f11_31', obs)
    white=[100.962 100.00 64.350];
elseif strcmp('user', obs)
    white=xyzw;
else
   disp('unknown option obs'); 
   disp('use d65_64 for D65 and 1964 observer'); return;
end

lab = zeros(size(xyz,1),3);  

fx = zeros(size(xyz,1),3);
for i=1:3
    index = (xyz(:,i)/white(i) > (6/29)^3);
    fx(:,i) = fx(:,i) + index.*(xyz(:,i)/white(i)).^(1/3);   
    fx(:,i) = fx(:,i) + (1-index).*((841/108)*(xyz(:,i)/white(i)) + 4/29);   
end

lab(:,1)=116*fx(:,2)-16;
lab(:,2) = 500*(fx(:,1)-fx(:,2));
lab(:,3) = 200*(fx(:,2)-fx(:,3));

end
% ====================================================
% *** END FUNCTION xyz2lab
% ====================================================















