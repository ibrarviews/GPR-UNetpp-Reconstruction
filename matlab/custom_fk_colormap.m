function cmap = custom_fk_colormap(N)
% Colormap matching reference style:
% dark navy -> blue -> cyan -> green -> yellow -> red
% Blue background with warm energy concentrations
    colors = [
        0.00  0.00  0.15;   % very dark navy  (noise floor -60 dB)
        0.00  0.10  0.45;   % dark blue       (-50 dB)
        0.05  0.30  0.75;   % medium blue     (-40 dB)
        0.00  0.60  0.90;   % sky blue        (-30 dB)
        0.00  0.85  0.85;   % cyan            (-20 dB)
        0.20  0.95  0.40;   % green           (-15 dB)
        0.95  0.95  0.00;   % yellow          (-10 dB)
        1.00  0.55  0.00;   % orange          (-5 dB)
        0.95  0.05  0.05;   % red             (-2 dB)
        1.00  1.00  1.00;   % white           (0 dB peak)
    ];
    x_in  = linspace(0, 1, size(colors, 1));
    x_out = linspace(0, 1, N);
    cmap  = interp1(x_in, colors, x_out);
    cmap  = max(0, min(1, cmap));
end
