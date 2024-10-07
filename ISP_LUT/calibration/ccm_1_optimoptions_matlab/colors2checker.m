function [img, keypoints, hfig] = colors2checker(color_groups, varargin)
%%
% COLORS2CHECKER visualizes color responses by drawing a color checker.
%
% USAGE:
% [img, hfig] = colorcheckervis(color_groups, 'param', value, ...);
%
% INPUTS:
% color_groups:      color responses to be drawn. If color_groups is a Nx3
%                    matrix, color in each row will be drawn as a patch in
%                    the color checker. If color_groups is a cell array
%                    containing multiple Nx3 matrices (color groups),
%                    colors with the same row index but from different
%                    groups will be compared within a color patch. The
%                    numbers of colors can be different for different
%                    groups (but not recommended), in this case part of the
%                    patch will be filled with BLANK_SAMPLE_COLOR if that
%                    response is missing. The supported maximum number of
%                    color groups is 5.
%
% OPTIONAL PARAMETERS:
% legend:            names of different color groups. It must be a cell
%                    array containing character arrays. If there is only
%                    one color group, no legend will be displayed. (default
%                    = {})
% direction:         the direction in which the colors are indexed.
%                    'row' (default) | 'column'
% layout:            the numbers of patches in y and x directions in the
%                    color checker, where x*y must be equal to the (max)
%                    number of colors in color groups. For example, [4, 6]
%                    for the classic color checker. If this parameter is
%                    not given, the layout of color checker will be
%                    determined automatically. (default = [])
% squaresize:        the size of each color patch in pixel (default = 200)
% parent:            axes object that specify the parent axes of the image.
%                    If not given, a new figure and a axes will be created.
%                    (default = [])
% show:              boolean value to determine whether to show the
%                    generated color checker. If set to false, only return
%                    the color checker image. (default = true)
%
% OUTPUTS:
% img:               the color checker image.
% keypoints:         key points of color patches in [x, y] form.
% hfig:              the figure handle.
%
% Copyright
% Qiu Jueqin - Feb, 2019

% parse the input parameters
param = parseInput(varargin{:});

% border between patches
BORDER = round(param.squaresize / 8);
if mod(BORDER, 2) ~= 0
    BORDER = BORDER + 1;
end
BORDER_HALF = BORDER/2;
BORDER_COLOR = [.18, .18, .18];
BLANK_SAMPLE_COLOR = [1, 1, 1];

if ~iscell(color_groups)
    color_groups = {color_groups};
end
groups_num = numel(color_groups);
assert(groups_num >= 1 && groups_num <= 5,...
       'The supported maximum number of color groups is 5.');

% numbers of colors for different groups
sample_nums = cellfun(@(x) size(x, 1), color_groups);
max_sample_num = max(sample_nums);
assert(max_sample_num <= 1024,...
       'The number of samples exceeds the maximum limit (1024).');

% numbers of patches in y and x dirrections in the color checker
if isempty(param.layout)
    param.layout = factor2(max_sample_num);
else
    assert(numel(param.layout)==2 && prod(param.layout) == max_sample_num,...
           ['''layout'' should be a 2-element vector [r, c], ',...
            'where r*c must be equal to the number of color samples.']);
end

% draw the color checker image
img = cell(param.layout);
keypoints = cell(param.layout);
for j = 1:max_sample_num
    colors = zeros(groups_num, 3);
    for i = 1:groups_num
        if j <= size(color_groups{i}, 1)
            colors(i, :) = color_groups{i}(j, :);
        else
            colors(i, :) = BLANK_SAMPLE_COLOR;
        end
    end
    % indexing direction
    switch param.direction
        case 'row'
            [col, row] = ind2sub(param.layout(end:-1:1), j);
        case 'column'
            [row, col] = ind2sub(param.layout, j);
        otherwise
            error('''direction'' parameter can only be ''row'' or ''column''.');
    end
    % draw a single patch
    img{row, col} = color2square(colors, param.squaresize, BORDER, BORDER_COLOR);
    keypoints{row, col} = [BORDER_HALF + 1, BORDER_HALF + 1;...
                           BORDER_HALF + param.squaresize, BORDER_HALF + 1;...
                           BORDER_HALF + 1, BORDER_HALF+param.squaresize;...
                           BORDER_HALF + param.squaresize, BORDER_HALF+param.squaresize] + ...
                           [(col-1)*(param.squaresize+2*BORDER_HALF), (row-1)*(param.squaresize+2*BORDER_HALF)];
end

img = cell2mat(img);
keypoints = keypoints';
keypoints = cell2mat(keypoints(:));

% add a (half) border
[h, w, ~] = size(img);
column_border = repmat(reshape(BORDER_COLOR, 1, 1, 3),...
                       h, BORDER_HALF, 1);
row_border = repmat(reshape(BORDER_COLOR, 1, 1, 3),...
                    BORDER_HALF, w + BORDER, 1);
img = [column_border, img, column_border];
img = [row_border; img; row_border];
keypoints = keypoints + [BORDER_HALF, BORDER_HALF];
[h, w, ~] = size(img);
keypoints = [1, 1;...
             w, 1;...
             1, h;...
             w, h;...
             keypoints]; % 4 corners on the outer box

% legend
if ~iscell(param.legend)
    param.legend = {param.legend};
end
assert(numel(param.legend) <= groups_num,...
       'The number of legend exceeds the number of color groups.');

switch groups_num
    case 1
        captions = {''};
        img_height = 0.85; % adjust the image height compared to the figure
    case 2
        captions = {'Center', 'Periphery'};
        img_height = 0.77;
    case 3
        captions = {'Center', 'Left', 'Right'};
        img_height = 0.73;
    case 4
        captions = {'Top left', 'Top right', 'Bottom left', 'Bottom right'};
        img_height = 0.69;
    case 5
        captions = {'Center', 'Top left', 'Top right', 'Bottom left', 'Bottom right'};
        img_height = 0.65;
end

legend_str = cell(1, groups_num);
for i = 1:groups_num
    if i > numel(param.legend)
        param.legend{i} = sprintf('color group %d', i); % default group name
    end
    legend_str{i} = sprintf('\\bf %s: \\rm %s', captions{i}, param.legend{i});  
end
legend_str = strjoin(legend_str, '\n');

if param.show == true
    if isempty(param.parent)
        try
            hfig = figureFullScreen('color', 'w');
        catch
            hfig = figure('color', 'w');
        end
        hax = axes(hfig, 'Position', [0.05, 0.95-img_height, 0.9, img_height]);
    else
        hax = param.parent;
    end
    imshow(img, 'Parent', hax);
    if groups_num > 1
        xlabel(legend_str, 'fontname', 'times new roman', 'fontsize', 18);
    end
else
    hfig = [];
end

end


function square = color2square(colors, squaresize, border, border_color)
%%
% COLOR2SQUARE draws a single color patch
% 'colors' is a M*3 matrix, in which each row corresponds to a color group
if nargin < 4
    border_color = [0, 0, 0];
end
[colors_num, ch_num] = size(colors);
assert(colors_num >= 1 && colors_num <= 5,...
       'The number of colors to be compared within a square patch should be in the range 1-5.');
assert(ch_num == 3, 'Only RGB colors are supported.');
square = zeros(squaresize,...
               squaresize,...
               3);
middle = round(squaresize / 2);
quarter = round(squaresize / 4);
for ch = 1:3
    switch colors_num
        case 1
            square(:, :, ch) = colors(1, ch);
        case 2
            square(:, :, ch) = colors(2, ch);
            square(quarter:squaresize-quarter, quarter:squaresize-quarter, ch) = colors(1, ch);
        case 3
            square(:, 1:middle, ch) = colors(2, ch);
            square(:, middle+1:end, ch) = colors(3, ch);
            square(quarter:squaresize-quarter, quarter:squaresize-quarter, ch) = colors(1, ch);
        case 4
            square(1:middle, 1:middle, ch) = colors(1, ch);
            square(1:middle, middle+1:end, ch) = colors(2, ch);
            square(middle+1:end, 1:middle, ch) = colors(3, ch);
            square(middle+1:end, middle+1:end, ch) = colors(4, ch);
        case 5
            square(1:middle, 1:middle, ch) = colors(2, ch);
            square(1:middle, middle+1:end, ch) = colors(3, ch);
            square(middle+1:end, 1:middle, ch) = colors(4, ch);
            square(middle+1:end, middle+1:end, ch) = colors(5, ch);
            square(quarter:squaresize-quarter, quarter:squaresize-quarter, ch) = colors(1, ch);
    end
end

% add the border
tmp = square;
square = repmat(reshape(border_color, 1, 1, 3),...
                squaresize + border,...
                squaresize + border,...
                1);
square(round(border/2)+[1:squaresize],...
       round(border/2)+[1:squaresize],...
       :) = tmp;
end


function fct2 = factor2(num)
%%
% factorize num into two closest factors f1 and f2 (f1 < f2)
f1 = floor(sqrt(num/1.4)); % quotient >1 for a rectangle instead of square
while mod(num, f1) ~= 0
    f1 = f1 - 1;
end
fct2 = [f1, num/f1];
end


function param = parseInput(varargin)
%%
% parse inputs & return structure of parameters
parser = inputParser;
parser.PartialMatching = false;
parser.addParameter('direction', 'row', @ischar);
parser.addParameter('layout', [], @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.addParameter('legend', {}, @(x)validateattributes(x, {'char', 'cell'}, {}));
parser.addParameter('parent', [], @ishandle);
parser.addParameter('show', true, @islogical);
parser.addParameter('squaresize', 200, @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.parse(varargin{:});
param = parser.Results;
% check the params
assert(param.squaresize >= 50,...
       'square size must be greater than 50.');
if ~isempty(param.parent)
    param.show = true;
end
end