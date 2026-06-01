function cmap = redwhiteblue(N)
% REDWHITEBLUE  Diverging red-white-blue colormap for GPR B-scans.
%   Positive amplitudes = red, zero = white, negative = blue.
%   Classic in GPR and seismic literature.
%
%   cmap = redwhiteblue(N)   returns Nx3 colormap matrix

    if nargin < 1, N = 256; end

    colors = [
        0.70  0.05  0.05;   % dark red    (strong positive)
        0.90  0.30  0.20;   % red
        0.97  0.60  0.50;   % light red
        1.00  1.00  1.00;   % white       (zero crossing)
        0.55  0.75  0.95;   % light blue
        0.20  0.45  0.80;   % blue
        0.05  0.15  0.60;   % dark blue   (strong negative)
    ];

    x_in  = linspace(0, 1, size(colors,1));
    x_out = linspace(0, 1, N);
    cmap  = interp1(x_in, colors, x_out);
    cmap  = max(0, min(1, cmap));
end
