function [matrix, scale, predicted_responses, errs] = ...
         ccmtrain(camera_responses,...
                  target_responses,...
                  varargin)
% CCMTRAIN calculates the optimal color correction matrix between the
% camera responses and the target responses by minimizing the nonlinear
% loss function. Both polynomial and root-polynomial color correction
% models are supported.
%
% USAGE:
% [M, scale, predicted_responses, errs_train] = ccmtrain(camera_responses,...
%                                                        target_responses,...
%                                                        'param', value, ...);
%
% INPUTS:
% camera_responses:  Nx3 camera linear RGB responses in the range 0-1, with
%                    darkness level subtracted. (can also be XYZ responses
%                    in some particular cases)
% target_responses:  Nx3 target linear RGB/XYZ responses in the range 0-1.
%
% OPTIONAL PARAMETERS:
% loss:              loss function to be minimized.
%                    'mse' | 'ciede00' (default) | 'ciede94' |
%                    'ciedelab' | 'cmcde'
% model:             color correction model, based on which the camera
%                    responses will be expanded.
%                    'linear3x3' (default) | 'root6x3' | 'root13x3' |
%                    'poly4x3' | 'poly6x3' | 'poly7x3' | 'poly9x3'
% omitlightness:     boolean value. Set to true to omit lightness component
%                    when optimizing the color correction matrix. This
%                    option will be useful if the camera responses are in a
%                    different range from the target responses due to the
%                    exposure difference. (default = false)
% bias:              boolean value. Set to true to add 1's to the final
%                    column of the expanded camera responses, e.g., 
%                    [R, G, B, R*G*B, 1] for 'poly4x3' model. Note: adding
%                    a bias to the camera responses will no longer ensure
%                    the independency of camera exposure, so do NOT enable
%                    this option if the responses were captured under
%                    different exposure parameters for training and
%                    validation. (default = false)
% allowscale:        boolean value. If set to true, the camera responses
%                    will be first scaled by a factor such that the mse
%                    between camera's G values and target G (or Y, 
%                    depending on 'targetcolorspace') values is minimized.
%                    This option will be useful if the camera responses are
%                    in a different range from the target responses. When 
%                    this option is enabled, 'omitlightness' option will be 
%                    false. (default = true)
% targetcolorspace:  specify the color space for the target responses.
%                    'sRGB' (default. It must be LINEAR sRGB) | 'XYZ'
% preservewhite:     boolean value. If set to true, the white point [1, 1,
%                    1] will be preserved after color correction as
%                    WHITEPOINT, i.e., [1, 1, 1,...] * matrix = WHITEPOINT.
%                    This constraint will be useful if the camera responses
%                    have been white-balanced, as the unconstrained
%                    optimization may cause the white point deviating from
%                    the target value. See demo2.m for example. (default =
%                    false)
% whitepoint:        1x3 white point vector which is to be preserved after
%                    color correction. (default = [])
% observer:          specify the standard colorimetric observer functions
%                    when converting XYZ to L*a*b* or vice versa.
%                    '1931' (default) | '1964'
% refilluminant:     specify the reference illuminant when converting XYZ
%                    to L*a*b* or vice versa.
%                    'A' | 'C' | 'D50' | 'D55' | 'D65' (default) |
%                    'D75' | 'F2' | 'F11'
% metric:            color difference metrics. Same as 'loss', but only for
%                    evaluation. It should be a char or a cell containing
%                    one or more of following metrics.
%                    'mse' | 'ciede00' | 'ciede94' | 'ciedelab' | 'cmcde'
%                    (default = {'ciede00', 'ciedelab'})
% weights:           weight coefficient for samples. It must be a 1xN
%                    vector, where N is the number of samples. (default = 
%                    [1, 1, ..., 1])
%
% OUTPUTS:
% matrix:            the optimal color correction matrix
% scale:             the optimal scaling factor
% predicted_responses:  the color corrected responses predicted by 'scale'
%                       and 'matrix', i.e., predicted_responses =
%                       (scale * expanded_camera_responses) * matrix
% errs:              a structure array containing color differences
%                    on training sample specified by 'metric'
%
% Copyright
% Qiu Jueqin - Feb, 2019

% parse and check the input parameters 
param = parseInput(varargin{:});
param = paramCheck(param);

% check the inputs
N = size(camera_responses, 1);
assert(isequal(size(camera_responses), size(target_responses)),...
       'The numbers of test and target samples do not match.');
assert(size(camera_responses, 2) == 3,...
       'Both test and target responses must be Nx3 matrices.');
% assert(all(camera_responses >= 0 & camera_responses <= 1, 'all'),...
%        'Test responses must be in the range of [0, 1]. Normalize them before running optimization.');
if ~isempty(param.weights)
    assert(numel(param.weights) == N,...
           'The number of weights does not match the number of test samples.');
end

% determine the number of terms
term_num = regexp(param.model, '(\d{1,2})x', 'tokens');
term_num = str2double(term_num{1}{1});
if param.bias == true
    term_num = term_num + 1;
end
if N <= term_num
    error('The number of sample must be greater than %d for model ''%s''.',...
          term_num, param.model);
end

% obs will be used to determine the condition for the conversion between
% XYZ values and L*a*b* values. See lab2xyz_.m and xyz2lab_.m for details.
switch param.observer
    case '1931'
        obs = [lower(param.refilluminant), '_31'];
    case '1964'
        obs = [lower(param.refilluminant), '_64'];
end

% normalize the weights
if ~isempty(param.weights)
    param.weights = N * param.weights / sum(param.weights);
else
    param.weights = ones(N, 1);
end

% loss function handle
lossfun = eval(['@', param.loss]);

% print info
paramPrint(param);
if param.preservewhite
    fprintf(['Note: white point [1, 1, 1] will be preserved as [%.3G, %.3G, %.3G]. ',...
             'Make sure \nthat the camera RGB responses have been correctly white balanced.\n\n'],...
             param.whitepoint);
end

% convert target responses to L*a*b* values
switch lower(param.targetcolorspace)
    case 'srgb'
        target_xyz = linsrgb2xyz(target_responses);
        target_lab = xyz2lab_(100*target_xyz, obs);
    case 'xyz'
        target_lab = xyz2lab_(100*target_responses, obs);
end

% scaling
if param.allowscale == true
    scale = fminbnd(@(x) mean( param.weights .* (x*camera_responses(:, 2) - target_responses(:, 2)).^2 ),...
                    0, 1E3); % minimize the mse between camera's Green values and target Green/Y values
    scale = min(scale, 1/max(camera_responses(:)));
    camera_responses = scale * camera_responses;
    camera_responses = max(min(camera_responses, 1), 0);
else
    scale = 1;
end

% expand the camera responses
expanded_camera_responses = response_expand(camera_responses, param.model, param.bias);
if param.preservewhite == true
    expanded_white_point = response_expand([1, 1, 1], param.model, param.bias);
end

% matrix calculation
% init: weighted least squares
matrix0 = (expanded_camera_responses' * diag(param.weights) * expanded_camera_responses)^(-1) *...
          expanded_camera_responses' * diag(param.weights) * target_responses;

switch lower(param.loss)
    case 'mse'
        matrix = matrix0;
    otherwise % nonlinear optimization
        matrix = @(x) reshape(x, term_num, 3);
        predicted_responses = @(x) expanded_camera_responses * matrix(x);

        switch lower(param.targetcolorspace)
            case 'srgb'
                predicted_xyz = @(x) linsrgb2xyz(predicted_responses(x));
                predicted_lab = @(x) xyz2lab_(100*predicted_xyz(x), obs);
            case 'xyz'
                predicted_lab = @(x) xyz2lab_(100*predicted_responses(x), obs);
        end

        errs = @(x) lossfun(predicted_lab(x),...
                            target_lab,...
                            param.omitlightness);
        errs = @(x) param.weights .* errs(x);
        costfun = @(x) mean(errs(x));

        % white point preserving constraint
        if param.preservewhite == true
            Aeq = blkdiag(expanded_white_point,...
                          expanded_white_point,...
                          expanded_white_point);
            beq = param.whitepoint;
        else
            Aeq = []; beq = [];
        end

        options = optimoptions(@fmincon,...
                               'MaxFunctionEvaluations', 10000,...
                               'MaxIterations',2000,...
                               'Display','iter',...
                               'Algorithm','sqp',...
                               'PlotFcns',[]);
        matrix = fmincon(costfun, matrix0(:), [], [], Aeq, beq, [], [], [], options);
        matrix = reshape(matrix, term_num, 3);
end

% validation

% camera_responses has been scaled in L160, so set scale param to 1 here
predicted_responses = ccmapply(camera_responses,...
                               param.model,...
                               matrix,...
                               1);
switch lower(param.targetcolorspace)
	case 'srgb'
        predicted_xyz = linsrgb2xyz(predicted_responses);
        predicted_lab = xyz2lab_(100*predicted_xyz, obs);
    case 'xyz'
        predicted_lab = xyz2lab_(100*predicted_responses, obs);
end

clear errs
disp('# Color correction training results:');
disp('=================================================================');
for i = 1:numel(param.metric)
    switch lower(param.metric{i})
        case 'mse'
            errs.(param.metric{i}) = mean((predicted_responses - target_responses).^2, 2);
        otherwise
            lossfun = eval(['@', param.metric{i}]); % metric handle
            errs.(param.metric{i}) = lossfun(predicted_lab,...
                                             target_lab,...
                                             param.omitlightness);
    end
	fprintf('%s errors: %.4G (avg), %.4G (med), %.4G (max, #%d)\n',...
            param.metric{i},...
            mean(errs.(param.metric{i})),...
            median(errs.(param.metric{i})),...
            max(errs.(param.metric{i})),...
            find(errs.(param.metric{i}) == max(errs.(param.metric{i}))));
end
if param.omitlightness == true
    disp('# (lightness component has been omitted)');
end
disp('=================================================================');
end


function param = parseInput(varargin)
% parse inputs & return structure of parameters
parser = inputParser;
parser.addParameter('allowscale', false, @(x)islogical(x));
parser.addParameter('bias', false, @(x)islogical(x));
parser.addParameter('loss', 'ciede00', @(x)ischar(x)); % for optimization
parser.addParameter('metric', {'ciede00', 'ciedelab'}, @(x)validateattributes(x, {'char', 'cell'}, {})); % for evaluation
parser.addParameter('model', 'linear3x3', @(x)ischar(x));
parser.addParameter('observer', '1931', @(x)ischar(x));
parser.addParameter('omitlightness', false, @(x)islogical(x));
parser.addParameter('preservewhite', false, @(x)islogical(x));
parser.addParameter('refilluminant', 'D65', @(x)ischar(x));
parser.addParameter('targetcolorspace', 'sRGB');
parser.addParameter('weights', [], @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.addParameter('whitepoint', [], @(x)validateattributes(x, {'numeric'}, {'positive'}));
parser.parse(varargin{:});
param = parser.Results;
end


function param = paramCheck(param)
% check the parameters

% check the color correction model
models_list = {'linear3x3',...
               'root6x3', 'root13x3',...
               'poly4x3', 'poly6x3', 'poly7x3', 'poly9x3'};
if ~ismember(lower(param.model), models_list)
    error('%s is not a valid color correction model. Only following models are supported:\n%s',...
          param.model, strjoin(models_list, ' | '));
end
if param.bias == true
	warning(['adding bias will no longer ensure the independency of camera exposure. ',...
             'Set ''bias'' to false if the responses were captured under different ',...
             'exposure parameters for training and validation.']);
end

% check the loss function
metrics_list = {'mse', 'ciede00', 'ciede94', 'ciedelab', 'cmcde'};
if ~ismember(lower(param.loss), metrics_list)
    error('%s is not a valid loss function. Only following losses are supported:\n%s',...
          param.loss, strjoin(metrics_list, ' | '));
end

% check the metrics
if isempty(param.metric)
    param.metric = {param.loss};
end
if ~iscell(param.metric)
    param.metric = {param.metric};
end
for i = 1:numel(param.metric)
    if ~ismember(lower(param.metric{i}), metrics_list)
        error('%s is not a valid metric. Only following metrics are supported:\n%s',...
              param.metric{i}, strjoin(metrics_list, ' | '));
    end
end

% check the reference illuminants
refilluminants_list = {'A', 'C', 'D50', 'D55', 'D65', 'D75', 'F2', 'F11'};
if ~ismember(upper(param.refilluminant), refilluminants_list)
    error('%s is not a valid reference illuminant. Only following illuminants are supported:\n%s',...
          param.refilluminants, strjoin(refilluminants_list, ' | '));
end

% check the standard observer
stdobserver_list = {'1931', '1964'};
if ~ismember(upper(param.observer), stdobserver_list)
    error('%s is not a valid standard observer function. Only following observers are supported:\n%s',...
          param.observer, strjoin(stdobserver_list, ' | '));
end

% check the white point
if ~isempty(param.whitepoint)
    assert(numel(param.whitepoint) == 3,...
           'The value of ''whitepoint'' must be a 1x3 vector.');
    param.preservewhite = true;
elseif param.preservewhite == true
    param.whitepoint = whitepoint('d65');
end
if param.preservewhite == true &&...
      (~ismember(lower(param.model), {'linear3x3', 'root6x3', 'root13x3'}) ||...
      strcmpi(param.loss, 'mse') ||...
      param.bias == true)
    warning(sprintf(['white point preserving will be deactivated when one of following cases occurs: \n',...
                     '* selecting one of ''poly4x3'', ''poly6x3'', ''poly7x3'', and ''poly9x3'' models; \n',...
                     '* using ''mse'' loss function; \n',...
                     '* setting ''bias'' to be true.']));
    param.preservewhite = false;
    param.whitepoint = [];
end

% check the scaling
if param.allowscale == true
    if param.omitlightness == true
        warning('lightness will be included in optimization when setting ''allowscale'' to be true.');
        param.omitlightness = false;
    end
elseif param.omitlightness == false
    warning(['including lightness has risk of making the optimization hard to converge. ',...
            'You can solve it by setting ''omitlightnss'' to be true ',...
            'or by setting ''allowscale'' to be true ',...
            'or by carefully scaling the camera RGB responses in advance.']);
end
end


function paramPrint(param)
% make format pretty
attr_idx = [3, 5, 7, 2, 1, 10, 8, 12, 6, 9, 4, 11];
param.weights = sprintf('[%.2G, ..., %.2G]',...
                        param.weights(1), param.weights(end));
if isempty(param.whitepoint)
    param.whitepoint = 'N/A';
else
    param.whitepoint = sprintf('[%.3G, %.3G, %.3G]',...
                               param.whitepoint);
end
if strcmpi(param.targetcolorspace, 'sRGB')
    param.targetcolorspace = 'sRGB (linear)';
elseif strcmpi(param.targetcolorspace, 'XYZ')    
    param.targetcolorspace = 'CIE1931 XYZ';
end
if numel(param.metric) > 1
    param.metric = strjoin(param.metric, ', ');
else
    param.metric = param.metric{1};
end
disp('Color correction training parameters:')
disp('=================================================================');
field_names = fieldnames(param);
field_name_dict.allowscale = 'Allow scaling camera responses';
field_name_dict.bias = 'Add bias (offset)';
field_name_dict.loss = 'Loss function';
field_name_dict.metric = 'Color difference metrics';
field_name_dict.model = 'Color correction model';
field_name_dict.observer = 'CIE standard colorimetric observer';
field_name_dict.omitlightness = 'Omit the lightness';
field_name_dict.preservewhite = 'Preserve the white point';
field_name_dict.refilluminant = 'Reference illuminant';
field_name_dict.targetcolorspace = 'Color space of the target responses';
field_name_dict.weights = 'Sample weights';
field_name_dict.whitepoint = 'White point';

for i = attr_idx
    len = fprintf('%s:',field_name_dict.(field_names{i}));
    fprintf(repmat(' ', 1, 42-len));
    fprintf('%s\n', string(param.(field_names{i})));
end
disp('=================================================================');
end
