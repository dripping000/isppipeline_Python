function [colors, roi_rects] = checker2colors(checker_img, layout, varargin)
%%
% CHECKER2COLORS extracts color responses from an image containing color
% checker. Two modes, 'drag' and 'click', are supported. ('auto' mode is to
% be done in the future)
%
% Note: for applications where high accuracies are required, you'd better
%       do some correction for the color checker image first, e.g., spatial
%       non-uniformity correction, noise reduction, etc.
%
% USAGE:
% [colors, roi_rects] = checker2colors(checker_img, layout,...
%                                      'param', value, ...);
%
% INPUTS:
% checker_img:       image containing color checker. It can also be a
%                    tensor with more than 3 channels, e.g., a
%                    R/G/B/IR image or a hyperspectral image.
% layout:            the numbers of patches in y and x directions in the
%                    color checker. For example, [4, 6] for the classic
%                    color checker. Note that this parameter is independent
%                    on the placement direction (landscape or portrait) of
%                    the color checker in the image.
%
% OPTIONAL PARAMETERS:
% mode:              method to locate the color checker. If 'drag' is
%                    chosen, you will be asked to drag a rectangle in the
%                    image to exactly cover the 4 vertices of the color
%                    checker; if 'click' is chosen, you will be asked to
%                    successively click 4 vertices in case that the color
%                    checker in the image has suffered from projective 
%                    transformation. In the 'click' mode, the first point
%                    clicked by user will be treated as the upper-left
%                    corner of the color checker. So if the color checker
%                    is placed upsidedown, you have to use 'click' to
%                    extract patches in the correct order.
%                    'drag' (default) | 'click'
% roi:               explicitly specify the coordinates of the roi
%                    rectangles. Must be a Nx4 matrix, where N = layout(1)
%                    * layout(2) and each row is in [x_begin, y_begin,
%                    width, height] form. This parameter is useful when in
%                    a batch extraction scenario, where user is expecting
%                    toextract responses from multiple images with the
%                    fixed color checker positions. For the first image,
%                    use [colors_1, roi] = checker2colors(checker_img,
%                    layout), and for the others, use 
%                    colors_N = checker2colors(checker_img, layout, 'roi',
%                    roi). If 'roi' is given, 'method', 'roisize', and
%                    'direction' parameters will be invalid, but you can
%                    still set 'allowadjust' to true to adjust the ROI
%                    positions. (default = [])
% allowadjust:       boolean value. If set to true, user will be allowed to
%                    adjust the ROI positions in the image after locating
%                    the vertices of color checker. ROI is where the color
%                    responses are to be extracted from. (default = false)
% roisize:           size of the square ROI. If not given, it will be
%                    automatically determined to be about 40% of the patch
%                    size (inferred by both 'layout' and the color checker
%                    region you chose). (default = [])
% direction:         the direction in which the patches are indexed.
%                    'row' (default) | 'column'
% scale:             the scaling factor to brighten the image. This
%                    parameter is uselful for some raw images that are
%                    darker then normal gamma corrected images. (default =
%                    1)
% gamma:             gamma correction for the image, same usage as 'scale'.
%                    (default = 1)
% show:              boolean value. Set to true to show the image as well
%                    as the ROI, otherwise to close the figure after
%                    vertices selection. You may wish to set it to false in
%                    the batch extraction cases. (default = true)
%
% OUTPUTS:
% colors:            N*C matrix of the extracted color responses, where N =
%                    layout(1) * layout(2) and C is the number of image
%                    channels (usually is 3).
% roi_rects:         coordinates of ROI rectangles, which can be cooperated
%                    with 'roi' parameter in the batch extraction cases.
%
% Copyright
% Qiu Jueqin - Feb, 2019

% parse the input parameters
param = parseInput(varargin{:});

% check the image data class
switch class(checker_img)
    case 'double'
        % do nothing
    case 'uint8'
        checker_img = double(checker_img) / (2^8 - 1);
    case 'uint16'
        checker_img = double(checker_img) / (2^16 - 1);
    otherwise
        error('The image class must be ''double'' or ''uint8'' or ''uint16''.');
end
if max(checker_img(:)) > 1 || max(checker_img(:)) < 0
    warning('the max intensity of the image exceeds the [0, 1] range.');
end
[height, width, depth] = size(checker_img);

% check the layout of color checker
assert(length(layout) == 2, '''layout'' must be a 2-element vector.');
N = prod(layout);

% get coordinates of the patches' centers
if isempty(param.roi)
    switch lower(param.mode)
        case 'drag'
            hint_str = 'Drag a rectangle to exactly cover the 4 corners of the color checker.';
            [~, hfig] = colorcheckerimshow(checker_img, hint_str, param.scale, param.gamma);
            try 
                hrect = drawrectangle('Label', 'Color Checker Rectangle');
                vertex_pts = bbox2points(hrect.Position);
            catch
                warning('Image Processing Toolbox is not found. Use ''getrect'' to get coordinate.');
                vertex_pts = bbox2points(getrect(hfig));
            end
        case 'click'
            hint_str = ['Click 4 points in the image to exactly cover the 4 corners of the color checker. ',...
                        'The first point you click will be stored as the upper-left vertex.'];
            [~, hfig] = colorcheckerimshow(checker_img, hint_str, param.scale, param.gamma);
            try 
                hrect = drawpolygon('Label', 'Color Checker Quadrangle');
                vertex_pts = hrect.Position;
            catch
                warning('Image Processing Toolbox is not found. Use ''ginput'' to get coordinate.');
                vertex_pts = ginput;
            end
        case 'auto'
            % to be done in the future
        otherwise
            error('''mode'' must be either ''drag'' or ''click''.');
    end
    % vertex_pts will be a 4x2 matrix in [A; B; C; D] order:
    % A - - - - B
    % |         |
    % D - - - - C
    vertex_pts = quadVertSort(vertex_pts);

    % determine the color checker placement direction in the image
    % (landscape or portrait)
    if vertex_pts(1,1) <= vertex_pts(3,1) &&...
       vertex_pts(1,2) <= vertex_pts(3,2)
        % normal placement: 1st patch in the upper-left corner
        img_rotation = 0;
        rotation_matrix = [1, 0; 0, 1];
        shift_vector = [0, 0];
    elseif vertex_pts(1,1) >= vertex_pts(3,1) &&...
           vertex_pts(1,2) <= vertex_pts(3,2)
        % 1st patch in the upper-right corner
        img_rotation = 90;
        rotation_matrix = [0, -1; 1, 0];
        shift_vector = [0, width];
        [height, width] = deal(width, height);
    elseif vertex_pts(1,1) >= vertex_pts(3,1) &&...
           vertex_pts(1,2) >= vertex_pts(3,2)
        % 1st patch in the bottom-right corner
        img_rotation = 180;
        rotation_matrix = [-1, 0; 0, -1];
        shift_vector = [width, height];
    elseif vertex_pts(1,1) <= vertex_pts(3,1) &&...
           vertex_pts(1,2) >= vertex_pts(3,2)
        % 1st patch in the bottom-left corner
        img_rotation = -90;
        rotation_matrix = [0, 1; -1, 0];
        shift_vector = [height, 0];
        [height, width] = deal(width, height);
    end
    
    % rotate image and transform the coordinates
    checker_img = imrotate(checker_img, img_rotation);
    vertex_pts = vertex_pts * rotation_matrix + shift_vector;
    
    % determine the roi size
    if isempty(param.roisize)
        roi_width = min(vertex_pts(2,1) - vertex_pts(1,1),...
                        vertex_pts(3,1) - vertex_pts(4,1)) / layout(2);
        roi_height = min(vertex_pts(4,2) - vertex_pts(1,2),...
                         vertex_pts(3,2) - vertex_pts(2,2)) / layout(1);
        param.roisize = ceil(0.4 * min(roi_width, roi_height));
    end
    radius = ceil(param.roisize/2);
    
    % each row in grid_pts coeresponds to the coordinate of the CENTER of
    % one color patch, in [x_center, y_center] form. grid_pts is in the
    % coordinate system of the rotated image
    grid_pts = vert2grids(vertex_pts, layout, param.direction);
    
    % clip
    grid_pts(:,1) = max(min(grid_pts(:,1), width), 1);
    grid_pts(:,2) = max(min(grid_pts(:,2), height), 1);
    
    % inversely rotate image and transform the coordinates back
    checker_img = imrotate(checker_img, -img_rotation);
    grid_pts = (grid_pts - shift_vector) * rotation_matrix^(-1);
             
    % determine the roi rectangles. roi_rects is in [x_begin, y_begin,
    % width, height] form
    roi_rects = [grid_pts - radius, repmat(param.roisize, N, 2)];
else
    assert(isequal(size(param.roi), [N, 4]),...
           '''points'' must be a Nx4 matrix, where N = layout(1) * layout(2).');
    roi_rects = param.roi;
end

if param.show
    if ~exist('hfig', 'var')
        [~, hfig] = colorcheckerimshow(checker_img,...
                                       'ROI specified by user',...
                                       param.scale,...
                                       param.gamma);
    end
    hroi = cell(N,1);
    for i = 1:N
        roi_rect = roi_rects(i, :);
        try
            hroi{i} = drawrectangle('Position', roi_rect,...
                                    'Label', num2str(i));
        catch
            hroi{i} = rectangle('Position', roi_rect);
        end
    end
end

% allow user to interactively adjust the positions of roi rectangles
if param.allowadjust
    set(hfig, 'Name', 'Adjust the ROI rectangles and click ''Continue'' to finish');
    fig_size = get(hfig, 'Position');
    hbutton = uicontrol(hfig,...
                       'Position', [fig_size(3)-120 20 100 40],...
                       'String', 'Continue',...
                       'Callback', @pushbutton_callback);
    uiwait(hfig);
    roi_rects = cell2mat(cellfun(@(x)x.Position, hroi, 'uniformoutput', false));
    commandwindow;
end

% determine the statistical way to extract responses from rois
getcolorfun = @median; % @mean is fine too

% extract responses from rois
colors = zeros(N, depth);
for i = 1:N
    % roi_rect_ is in [x_begin, y_begin, width, height] form
    roi_rect_ = round(roi_rects(i, :));
    roi_rect_ = max(roi_rect_, 1);
    roi_rect_(3) = min(roi_rect_(3), width - roi_rect_(1) - 1);
    roi_rect_(4) = min(roi_rect_(4), height - roi_rect_(2) - 1);
    
    roi = checker_img(roi_rect_(2) + [0:roi_rect_(4)],...
                      roi_rect_(1) + [0:roi_rect_(3)],...
                      :);
	colors(i,:) = squeeze(getcolorfun(getcolorfun(roi, 1), 2))';
end

if exist('hfig', 'var') && param.show == false
    close(hfig);
% else
%     commandwindow;
end
end


function ordered_pts = quadVertSort(pts)
%%
% function QUADVERTSORT reorders the 4 vertices of a quadrangle such
% that the output is in
% [A_x, A_y]
% [B_x, B_y]
% [C_x, C_y]
% [D_x, D_y]
% format, where A, B, C, D points are in clockwise order, and A is the
% first point given by user.
% For example, given 4 points from a rectangle in arbitrary order, the
% reordered output will be in the form  A - - - - B
%                                       |         |
%                                       D - - - - C
% (suppose you click the upper-left point first)

assert(isequal(size(pts), [4, 2]),...
       'Input must be a 4x2 matrix representing the 4 vertices of a quadrangle.');
center = mean(pts, 1);
% note that [x,y] is in image coordinate system 
% (y increases from top to bottom)
thetas = atan2(center(2) - pts(:,2),  pts(:,1) - center(1)); 
[~, orders] = sort(thetas, 'descend');
% shift circularly such that the upper-left vertex is in the first row
orders = circshift(orders, 1 - find(orders==1));
ordered_pts = pts(orders, :);
end


function [himg, hfig] = colorcheckerimshow(img, hint, scale, gamma)
%%
% show a color checker image with some hint message
try
    hfig = figureFullScreen('color', 'w', 'name', hint);
catch
    hfig = figure('color', 'w', 'name', hint);
end
himg = imshow((scale*img).^gamma,...
              'Border','tight',...
              'InitialMagnification','fit');
pause(0.1); % maximizing the figure has some delay
end


function grid_pts = vert2grids(vert_pts, layout, direction)
%%
% function VERT2GRIDS generates a set of grid coordinates inside a
% quadrangle, of which the 4 vertices are determined by vert_pts.

% create a rectangle of which the upper-left and bottom-right vertices
% coincide with the quadrangle
rect = [vert_pts(1, :),...
        vert_pts(3, 1) - vert_pts(1, 1),...
        vert_pts(3, 2) - vert_pts(1, 2)];
rect_pts = bbox2points(rect);

% find the projective transformation from the quadrangle to the rectangle
H = fitgeotrans(vert_pts, rect_pts, 'projective');

% grid inside the rectangle
[grid_x, grid_y] = meshgrid(linspace(rect_pts(1,1), rect_pts(3,1), 2*layout(2)+1),...
                            linspace(rect_pts(1,2), rect_pts(3,2), 2*layout(1)+1));
grid_x = grid_x(2:2:end, 2:2:end);
grid_y = grid_y(2:2:end, 2:2:end);
switch direction
    case 'row'
        grid_x = grid_x';
        grid_y = grid_y';
    case 'column'
        % do nothing
    otherwise
        error('''direction'' parameter can only be ''row'' or ''column''.');
end
grid_pts = [grid_x(:), grid_y(:)];
grid_pts = transformPointsInverse(H, grid_pts);
end


function pushbutton_callback(hObject, eventdata, handles)
%%
% callback function of the figure
uiresume;
set(hObject, 'Visible', 'off');
end


function param = parseInput(varargin)
%%
% parse inputs & return structure of parameters
parser = inputParser;
parser.PartialMatching = false;
parser.addParameter('allowadjust', false, @(x)islogical(x));
parser.addParameter('direction', 'row', @ischar);
parser.addParameter('mode', 'drag', @ischar);
parser.addParameter('roi', [], @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.addParameter('roisize', [], @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.addParameter('scale', 1, @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.addParameter('gamma', 1, @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.addParameter('show', true, @(x)islogical(x));
parser.parse(varargin{:});
param = parser.Results;
% check the params
if ~exist('drawrectangle', 'file') && param.allowadjust
    warning(['''allowadjust'' can not be truned on because function ''drawrectangle'' is not found. ',...
             'Image Processing Toolbox version R2018b or higher is required to create images.roi objects.']);
    param.allowadjust = false;
end
if param.allowadjust
    param.show = true;
end
end
