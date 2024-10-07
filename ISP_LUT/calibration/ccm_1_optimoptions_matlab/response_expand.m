function expanded = response_expand(responses, model, bias)
% RESPONSE_EXPAND expands RGB/XYZ responses into polynomial or
% root-polynomial terms as per chosen model.
%
% For example, Nx3 RGB responses  [R1, G1, B1]
%                                 [  ......  ]
%                                 [Rn, Gn, Bn]
% will be expanded into Nx4 terms [R1, G1, B1, R1*G1*B1]
%                                 [       ......       ]
%                                 [Rn, Gn, Bn, Rn*Gn*BN]
% if 'poly4x3' model was chosen
%
% INPUTS:
% responses:	Nx3 RGB/XYZ responses
% model:        color correction model.
%               'linear3x3' | 'root6x3' | 'root13x3' | 'poly4x3' | 
%               'poly6x3' | 'poly7x3' | 'poly9x3'
% bias:         boolean value. If true, add 1's to the final column of the
%               expanded responses as biases (offsets)
%
% OUTPUT:
% expanded:     NxM expanded RGB/XYZ responses, where M is determined by
%               the chosen model

[N, col_num] = size(responses);
assert(ismatrix(responses) && col_num == 3,...
       'Input must have size of Nx3.');
   
models_list = {'linear3x3',...
               'root6x3', 'root13x3',...
               'poly4x3', 'poly6x3', 'poly7x3', 'poly9x3'};
if ~ismember(lower(model), models_list)
    error('%s is not a valid color correction model. Only following models are supported:\n%s',...
          model, strjoin(model_list, ' | '));
end

R = responses(:, 1);
G = responses(:, 2);
B = responses(:, 3);
switch lower(model)
    case 'linear3x3'
        expanded = responses;
    case 'poly4x3'
        expanded = [responses,...
                    R.*G.*B]; % N*4
    case 'poly6x3'
        expanded = [responses,...
                    R.*G, R.*B, G.*B]; % N*6
    case 'poly7x3'
        expanded = [responses,...
                    R.*G, R.*B, G.*B,...
                    R.*G.*B]; % N*7
    case 'poly9x3'
        expanded = [responses,...
                    R.*G, R.*B, G.*B,...
                    R.^2, G.^2, B.^2]; % N*9
    case 'root6x3'
        expanded = [responses,...
                    sqrt(R.*G), sqrt(R.*B), sqrt(G.*B)]; % N*6
    case 'root13x3'
        expanded = [responses,...
                    sqrt(R.*G), sqrt(R.*B), sqrt(G.*B),...
                    (R.*G.^2).^(1/3), (R.*B.^2).^(1/3), (G.*B.^2).^(1/3),...
                    (G.*R.^2).^(1/3), (B.*R.^2).^(1/3), (B.*G.^2).^(1/3),...
                    (R.*G.*B).^(1/3)]; % N*13
end
if bias == true
    expanded = [expanded, ones(N, 1)];
end