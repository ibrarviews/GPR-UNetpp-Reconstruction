%% generate_FK_spectra_figure13.m
%  Generates Figure 13 — F-K spectra — JAG manuscript revision
%  Author: [Your name]
%
%  Required files in same folder:
%  original.png, pocs.png, unet.png, unetpp.png
%  load_bscan.m, compute_fk.m, custom_fk_colormap.m

close all; clear; clc;

%% FILES
file_original = 'original.png';
file_pocs     = 'pocs.png';
file_unet     = 'unet.png';
file_unetpp   = 'unetpp.png';

%% LOAD
D_orig   = load_bscan(file_original);
D_pocs   = load_bscan(file_pocs);
D_unet   = load_bscan(file_unet);
D_unetpp = load_bscan(file_unetpp);

[Nt, Nx] = size(D_orig);
D_pocs   = imresize(D_pocs,   [Nt Nx]);
D_unet   = imresize(D_unet,   [Nt Nx]);
D_unetpp = imresize(D_unetpp, [Nt Nx]);

fprintf('Image size: %d x %d\n', Nt, Nx);

%% PREPROCESS — remove column mean, apply 2D window
D_orig   = bsxfun(@minus, D_orig,   mean(D_orig));
D_pocs   = bsxfun(@minus, D_pocs,   mean(D_pocs));
D_unet   = bsxfun(@minus, D_unet,   mean(D_unet));
D_unetpp = bsxfun(@minus, D_unetpp, mean(D_unetpp));

% Light smoothing to reduce JPEG compression noise
smooth2d = @(D) imfilter(D, fspecial('gaussian',[3 3],0.8));
D_orig   = smooth2d(D_orig);
D_pocs   = smooth2d(D_pocs);
D_unet   = smooth2d(D_unet);
D_unetpp = smooth2d(D_unetpp);

win2d    = hanning(Nt) * hanning(Nx)';
D_orig   = D_orig   .* win2d;
D_pocs   = D_pocs   .* win2d;
D_unet   = D_unet   .* win2d;
D_unetpp = D_unetpp .* win2d;

%% COMPUTE F-K SPECTRA
FK_orig   = compute_fk(D_orig);
FK_pocs   = compute_fk(D_pocs);
FK_unet   = compute_fk(D_unet);
FK_unetpp = compute_fk(D_unetpp);

%% PHYSICAL AXES
T_total = 150e-9;   % 150 ns total time
X_total = 4.0;      % 4 m total distance
dt_img  = T_total / Nt;
dx_img  = X_total / Nx;

Nf      = floor(Nt/2);
f_axis  = (0:Nf-1) / (Nt * dt_img) / 1e6;              % MHz
kx_axis = fftshift((-Nx/2:Nx/2-1) / (Nx * dx_img));    % m^-1

fprintf('Freq: 0 to %.1f MHz\n', max(f_axis));
fprintf('Kx:   %.4f to %.4f m^-1\n', min(kx_axis), max(kx_axis));

%% dB NORMALISE — per panel
db_floor = -60;
global_max = max(FK_orig(:));   % reference = original
norm_db  = @(X) 20*log10(X ./ global_max + eps);
clamp    = @(X) max(min(X, 0), db_floor);

FK_orig_db   = clamp(norm_db(FK_orig));
FK_pocs_db   = clamp(norm_db(FK_pocs));
FK_unet_db   = clamp(norm_db(FK_unet));
FK_unetpp_db = clamp(norm_db(FK_unetpp));

%% FIGURE
fig = figure('Units','centimeters','Position',[2 2 22 18]);
fig.Color = 'white';

cmap   = custom_fk_colormap(512);
titles = {'(a) Original','(b) POCS','(c) U-Net','(d) U-Net++'};
datas  = {FK_orig_db, FK_pocs_db, FK_unet_db, FK_unetpp_db};

for p = 1:4
    ax = subplot(2, 2, p);
    imagesc(kx_axis, f_axis, datas{p});
    colormap(ax, cmap);
    caxis(ax, [db_floor 0]);
    axis tight;
    ax.YDir = 'normal';
    ylim([10 500]);      % start from 10 MHz to remove DC artifact
    xlim([-0.04 0.04]); % focus on coherent energy region

    if p == 3 || p == 4
        xlabel('Wavenumber (m^{-1})', 'FontSize', 9, 'FontName', 'Helvetica');
    end
    if p == 1 || p == 3
        ylabel('Frequency (MHz)', 'FontSize', 9, 'FontName', 'Helvetica');
    else
        ax.YTickLabel = {};
    end

    title(titles{p}, 'FontSize', 10, 'FontWeight', 'bold', ...
          'FontName', 'Helvetica');

    if p == 2 || p == 4
        cb = colorbar(ax, 'Location', 'eastoutside');
        cb.Label.String = 'Amplitude (dB)';
        cb.Label.FontSize = 8;
        cb.FontSize = 8;
        cb.Ticks = [-60 -50 -40 -30 -20 -10 0];
    end

    set(ax, 'FontSize', 9, 'FontName', 'Helvetica', ...
        'TickDir', 'out', 'LineWidth', 0.7, 'Box', 'on');
end

subplot(2,2,1); set(gca,'Position',[0.09 0.55 0.35 0.38]);
subplot(2,2,2); set(gca,'Position',[0.54 0.55 0.35 0.38]);
subplot(2,2,3); set(gca,'Position',[0.09 0.09 0.35 0.38]);
subplot(2,2,4); set(gca,'Position',[0.54 0.09 0.35 0.38]);

%% EXPORT
print(fig, 'Figure13_FK_spectra', '-dtiff', '-r600');
print(fig, 'Figure13_FK_spectra', '-dpdf',  '-r600');
fprintf('Saved: Figure13_FK_spectra.tif and .pdf\n');
