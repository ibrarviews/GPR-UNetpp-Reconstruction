function D = load_bscan(fname)
% Read grayscale PNG as double matrix, normalised to [-1, 1]
    img = imread(fname);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    D = double(img);
    D = 255 - D;              % invert: dark = high amplitude (GPR convention)
    D = (D / 127.5) - 1.0;   % normalise to [-1, 1]
end
