% ===================================================
% *** FUNCTION ciede00
% ***
% *** function [de,dl,dc,dh] = cie00de(lab1, lab2, omitlightness, sl, sc, sh)
% *** computes colour difference from CIELAB values 
% *** using CIEDE2000 formula
% *** inputs must be n by 3 matrices
% *** and contain L*, a* and b* values
% *** see also cielabde, cmcde, and cie94de
% 
% The lightness component will be omitted if omitlightness is true
%
% Modified based on the source version from 
% Computational Colour Science using MATLAB 2e
% https://www.mathworks.com/matlabcentral/fileexchange/
% 40640-computational-colour-science-using-matlab-2e
% ===================================================

function [de,dl,dc,dh] = ciede00(lab1,lab2,omitlightness,sl,sc,sh)

if (size(lab1,1)~=size(lab2,1))
   disp('inputs must be the same size'); return;   
end

if (size(lab1,2)~=3 || size(lab2,2)~=3)
   disp('inputs must be n by 3'); return;   
end
if (nargin<6)
    % disp('using default values of l:c')
    sl=1; sc=1; sh=1;
end
if nargin < 3
    omitlightness = false;
end

de = zeros(1,size(lab1,2));
dl = zeros(1,size(lab1,2));
dc = zeros(1,size(lab1,2));
dh = zeros(1,size(lab1,2));

% convert the cartesian a*b* to polar chroma and hue
[h1,c1] = cart2pol(lab1(:,2), lab1(:,3));
[h2,c2] = cart2pol(lab2(:,2), lab2(:,3));
h1 = h1*180/pi;
h2 = h2*180/pi;
meanC = (c2+c1)/2;

% compute G factor using the arithmetic mean chroma
G = 0.5 - 0.5*(((meanC.^7)./(meanC.^7 + 25^7)).^0.5);

% transform the a* values
lab1(:,2) = (1 + G).*lab1(:,2);
lab2(:,2) = (1 + G).*lab2(:,2);

% recompute the polar coordinates using the new a*
[h1,c1] = cart2pol(lab1(:,2), lab1(:,3));
[h2,c2] = cart2pol(lab2(:,2), lab2(:,3));
h1 = h1*180/pi;
h2 = h2*180/pi;

% compute the mean values for use later
meanC = (c2+c1)/2;
meanL = (lab2(:,1)+lab1(:,1))/2;

meanH = (h1+h2)/2;
% Identify positions for which abs hue diff exceeds 180 degrees 
meanH = meanH - (abs(h1-h2)>180)*180;
% rollover ones that come -ve
meanH = meanH + (meanH < 0)*360;
% Check if one of the chroma values is zero, in which case set 
% mean hue to the sum which is equivalent to other value
index = find(c1.*c2 == 0);
meanH(index) = h1(index)+h2(index);


% compute the basic delta values
dh = (h2-h1);
index = dh>180;
dh = (index).*(dh-360) + (1-index).*dh;
dh = 2*((c1.*c2).^0.5).*sin((dh/2)*pi/180);
dl = lab2(:,1)-lab1(:,1);
dc = c2-c1;

T = 1 - 0.17*cos((meanH-30)*pi/180) + 0.24*cos((2*meanH)*pi/180); 
T = T + 0.32*cos((3*meanH + 6)*pi/180) - 0.20*cos((4*meanH - 63)*pi/180);

dthe = 30*exp(-((meanH-275)/25).^2);
rc = 2*((meanC.^7)./(meanC.^7 + 25^7)).^0.5;
rt = -sin(2*dthe*pi/180).*rc;

Lweight = 1 + (0.015*(meanL-50).^2)./((20 + (meanL-50).^2).^0.5);
Cweight = 1 + 0.045*meanC;
Hweight = 1 + 0.015*meanC.*T;

dl = dl./(Lweight*sl);
dc = dc./(Cweight*sc);
dh = dh./(Hweight*sh);

%disp([G T Lweight Cweight Hweight rt])
if omitlightness
    de = sqrt(dc.^2 + dh.^2 + rt.*dc.*dh);
else
    de = sqrt(dl.^2 + dc.^2 + dh.^2 + rt.*dc.*dh);
end


