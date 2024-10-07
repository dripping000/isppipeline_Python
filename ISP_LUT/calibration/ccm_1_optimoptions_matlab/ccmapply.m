function predicted_responses = ccmapply(camera_responses,...
                                        model,...
                                        matrix,...
                                        scale)
% CCMAPPLY converts the camera responses into the corrected color responses,
% via a scaling factor and a color correction matrix.
%
% INPUTS:
% camera_responses:  Nx3 camera linear RGB responses to be validated, in
%                    the range 0-1, with darkness level subtracted. (can
%                    also be XYZ responses in some particular cases)
% model:             color correction model, based on which the camera
%                    responses will be expanded.
%                    'linear3x3' (default) | 'root6x3' | 'root13x3' |
%                    'poly4x3' | 'poly6x3' | 'poly7x3' | 'poly9x3'
% matrix:            color correction matrix, the size of which must match
%                    'model'.
% scale:             the scaling factor. Camera responses will be first
%                    scaled by this factor and then be expanded and
%                    multiplied. (default = 1)
%
% OUTPUTS:
% predicted_responses:  the color corrected responses predicted by 'scale'
%                       and 'matrix', i.e., predicted_responses =
%                       (scale * expanded_camera_responses) * matrix
%
% Copyright
% Qiu Jueqin - Feb, 2019

% check the inputs
assert(size(camera_responses, 2) == 3,...
       'Test responses must be a Nx3 matrix.');
assert(max(camera_responses(:)) <= 1 && min(camera_responses(:)) >= 0,...
       'Test responses must be in the range of [0, 1]. Normalize them before color correction.');

% check the color correction model
models_list = {'linear3x3',...
               'root6x3', 'root13x3',...
               'poly4x3', 'poly6x3', 'poly7x3', 'poly9x3'};
if ~ismember(lower(model), models_list)
    error('%s is not a valid color correction model. Only following models are supported:\n%s',...
          param.model, strjoin(models_list, ' | '));
end

% determine the number of terms
term_num = regexp(model, '(\d{1,2})x', 'tokens');
term_num = str2double(term_num{1}{1});

% check the matrix
if size(matrix, 1) == term_num + 1
    add_bias = true;
elseif size(matrix, 1) == term_num
    add_bias = false;
else
    error('The model ''%s'' and the matrix do not match.', model);
end

% scaling
if nargin < 4 || isempty(scale) || scale <= 0
    scale = 1;
end
camera_responses = scale * camera_responses;
camera_responses = max(min(camera_responses, 1), 0);

% expand the camera responses
expanded_camera_responses = response_expand(camera_responses, model, add_bias);

% color correction
predicted_responses = expanded_camera_responses * matrix;
