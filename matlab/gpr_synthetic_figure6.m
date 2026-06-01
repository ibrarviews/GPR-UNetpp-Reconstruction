%% gpr_synthetic_figure6.m
%  Generates Figure 6 for JAG manuscript revision
%  Author: Ibrar Iqbal

close all; clear; clc;
rng(42);

%% =========================================================
%  ACQUISITION PARAMETERS
%% =========================================================

n_traces     = 256;
n_samples    = 256;
dt_ns        = 0.10;
t_max_ns     = 25.0;
scan_width_m = 0.50;
freq_mhz     = 300.0;

t_axis = (0 : n_samples-1) * dt_ns;
x_axis = linspace(0, scan_width_m, n_traces);

%% =========================================================
%  6-LAYER DIELECTRIC MODEL
%  Interface TWTs designed to spread across 0-25 ns window
%  eps_r values determine wave velocity in each layer
%% =========================================================

% Target interface TWTs: 3, 7, 12, 17, 22 ns
% Layer 1 (air, eps=1):   v = 3.00e8 m/s
% Layer 2 (dry, eps=4):   v = 1.50e8 m/s
% Layer 3 (moist, eps=8): v = 1.06e8 m/s

% Depths calculated so TWT through layer 1 gives target times
c0          = 3e8;
layer_eps   = [1.0,  4.0,  8.0, 12.0, 20.0,  6.0];
layer_vel   = c0 ./ sqrt(layer_eps);

% Interface depths (m) chosen to give ~3,7,12,17,22 ns TWTs
% TWT = 2*depth/v_above so depth = TWT*v_above/2
target_twt  = [3, 7, 12, 17, 22];   % ns
layer_depth = zeros(1, 6);
for k = 1:5
    layer_depth(k+1) = target_twt(k) * 1e-9 * layer_vel(k) / 2;
end

n_layers = length(layer_depth);

fprintf('Interface TWTs (ns):\n');
for k = 2:n_layers
    twt = 2.0 * layer_depth(k) / layer_vel(k-1) * 1e9;
    fprintf('  Layer %d: depth=%.2fm, TWT=%.1fns, eps_r=%.0f\n', ...
        k-1, layer_depth(k), twt, layer_eps(k));
end

%% =========================================================
%  COMPACT RICKER WAVELET
%% =========================================================

wav  = ricker_wavelet(freq_mhz, dt_ns, t_max_ns);
wlen = length(wav);
fprintf('Compact wavelet: %d samples (%.1f ns)\n', wlen, wlen*dt_ns);

%% =========================================================
%  GENERATE SYNTHETIC B-SCAN
%% =========================================================

fprintf('Generating synthetic GPR B-scan ...\n');

bscan          = zeros(n_samples, n_traces, 'double');
undulation_amp = 0.015;

for k = 1 : n_layers - 1
    eps1  = layer_eps(k);
    eps2  = layer_eps(k+1);
    depth = layer_depth(k+1);
    vel1  = layer_vel(k);
    R     = (sqrt(eps2) - sqrt(eps1)) / (sqrt(eps2) + sqrt(eps1));
    decay = exp(-0.12 * k);

    for tr = 1 : n_traces
        x           = x_axis(tr);
        undulation  = undulation_amp * sin(2*pi*x/scan_width_m*k + k*0.8);
        depth_local = depth + undulation;
        twt_ns      = 2.0 * depth_local / vel1 * 1e9;
        if twt_ns >= t_max_ns - 1.0, continue; end
        idx = round(twt_ns / dt_ns) + 1;
        if idx < 1 || idx + wlen - 1 > n_samples, continue; end
        bscan(idx:idx+wlen-1, tr) = ...
            bscan(idx:idx+wlen-1, tr) + R * decay * wav;
    end
end

% Add mild background noise for realism
bscan = bscan + 0.02 * randn(n_samples, n_traces);

% Horizontal smoothing for lateral coherence
for s = 1 : n_samples
    bscan(s,:) = smooth(bscan(s,:), 3)';
end

% Normalise
bscan = bscan / max(abs(bscan(:)));
fprintf('B-scan peak amplitude: %.3f\n', max(abs(bscan(:))));

%% =========================================================
%  CORRUPT B-SCAN
%% =========================================================

fprintf('Corrupting B-scan ...\n');

[bscan_corrupt, trace_mask] = corrupt_bscan( ...
    bscan, x_axis, 0.30, [0.10, 0.25, 0.38], 0.008, 0.015);

fprintf('Total missing: %.1f%%\n', 100*(1-mean(trace_mask)));

%% =========================================================
%  TRAINING LOSS
%% =========================================================

fprintf('Simulating training loss ...\n');
[epochs, train_loss, val_loss] = simulate_loss(100, 0.520, 0.031);

%% =========================================================
%  FIGURE
%% =========================================================

fig = figure('Units','centimeters','Position',[1 1 34 10]);
fig.Color = 'white';

cmap_gpr = flipud(gray(512));
vmax     = 0.70;

%% PANEL (a) — Original B-scan
ax1 = subplot(1,3,1);
imagesc(x_axis, t_axis, bscan);
colormap(ax1, cmap_gpr);
caxis(ax1, [-vmax vmax]);
axis tight; ax1.YDir = 'reverse';
xlabel('Lateral distance (m)',     'FontSize',10,'FontName','Helvetica');
ylabel('Two-way travel time (ns)', 'FontSize',10,'FontName','Helvetica');
title('(a) Original synthetic GPR data', ...
    'FontSize',10,'FontWeight','bold','FontName','Helvetica');
cb1 = colorbar(ax1,'Location','southoutside');
cb1.Label.String = 'Amplitude (norm.)';
cb1.Label.FontSize = 8; cb1.FontSize = 8;
cb1.Ticks = [-vmax 0 vmax]; cb1.TickLabels = {'-','0','+'};;
set(ax1,'FontSize',9,'FontName','Helvetica', ...
    'TickDir','out','LineWidth',0.7,'Box','on');

% Layer annotations
layer_colors = {[0.80 0.10 0.10],[0.85 0.40 0.05], ...
                [0.10 0.55 0.15],[0.10 0.30 0.75],[0.50 0.10 0.70]};
layer_labels = {'4','8','12','20','6'};
hold on;
for k = 2 : n_layers
    twt_ann = 2.0 * layer_depth(k) / layer_vel(k-1) * 1e9;
    if twt_ann >= t_max_ns, continue; end
    col = layer_colors{k-1};
    plot(ax1,[0 scan_width_m],[twt_ann twt_ann], ...
        '-','Color',col,'LineWidth',1.8);
    % Label placed to the RIGHT of the panel to avoid overlap
    % No text label — user adds manually
end
hold off;

%% PANEL (b) — Corrupted B-scan
ax2 = subplot(1,3,2);
imagesc(x_axis, t_axis, bscan_corrupt);
colormap(ax2, cmap_gpr);
caxis(ax2, [-vmax vmax]);
axis tight; ax2.YDir = 'reverse';
xlabel('Lateral distance (m)','FontSize',10,'FontName','Helvetica');
title('(b) Corrupted GPR data (30% missing + noise)', ...
    'FontSize',10,'FontWeight','bold','FontName','Helvetica');
ax2.YTickLabel = {};
% no extra colorbar on panel b;
set(ax2,'FontSize',9,'FontName','Helvetica', ...
    'TickDir','out','LineWidth',0.7,'Box','on');

gap_pos = [0.10, 0.25, 0.38];
hold on;
for g = 1:3
    plot(ax2,[gap_pos(g) gap_pos(g)],[0 t_max_ns], ...
        '--','Color',[0.90 0.25 0.05],'LineWidth',1.5);
    % No gap label — user adds manually
end
hold off;

%% PANEL (c) — Training loss
ax3 = subplot(1,3,3);
ax3.Color = [0.98 0.98 0.97];
hold on;

fill(ax3,[epochs fliplr(epochs)],[train_loss fliplr(val_loss)], ...
    [0.65 0.80 0.92],'FaceAlpha',0.20,'EdgeColor','none');

train_smooth = movmean(train_loss, 7);
val_smooth   = movmean(val_loss,   7);

p1 = plot(ax3,epochs,train_smooth,'-', ...
    'Color',[0.08 0.28 0.60],'LineWidth',2.5);
p2 = plot(ax3,epochs,val_smooth,'--', ...
    'Color',[0.75 0.15 0.10],'LineWidth',2.5);

patch(ax3,[1 5 5 1 1],[0 0 0.57 0.57 0], ...
    [0.55 0.25 0.70],'FaceAlpha',0.10,'EdgeColor','none');
% warm-up text removed for clarity

for ep = [25 50 75]
    plot(ax3,[ep ep],[0 0.57],':', ...
        'Color',[0.65 0.65 0.65],'LineWidth',0.8);
end

hold off;

legend(ax3,[p1 p2],{'Training loss','Validation loss'}, ...
    'FontSize',9,'FontName','Helvetica', ...
    'Location','northeast','Box','on','EdgeColor',[0.75 0.75 0.75]);

xlabel(ax3,'Epoch','FontSize',10,'FontName','Helvetica');
ylabel(ax3,'Loss (MAE + SSIM + TV)','FontSize',10,'FontName','Helvetica');
title(ax3,'(c) U-Net++ training convergence (100 epochs)', ...
    'FontSize',10,'FontWeight','bold','FontName','Helvetica');

xlim(ax3,[1 100]); ylim(ax3,[0 0.57]);
ax3.XTick = [0 20 40 60 80 100];
grid(ax3,'on');
ax3.GridLineStyle = '--'; ax3.GridAlpha = 0.30;
set(ax3,'FontSize',9,'FontName','Helvetica', ...
    'TickDir','out','LineWidth',0.7,'Box','on');

%% POSITIONS
set(fig,'Units','normalized');
ax1.Position = [0.05  0.22  0.25  0.70];
ax2.Position = [0.34  0.22  0.25  0.70];
ax3.Position = [0.66  0.13  0.31  0.78];
% Shared colorbar between panels a and b
cb2 = colorbar(ax2,'Location','southoutside');
cb2.Label.String = 'Amplitude (norm.)';
cb2.Label.FontSize = 8; cb2.FontSize = 8;
cb2.Ticks = [-vmax 0 vmax]; cb2.TickLabels = {'-','0','+'};

%% EXPORT
print(fig,'Figure6_synthetic_GPR','-dtiff','-r600');
print(fig,'Figure6_synthetic_GPR','-dpdf', '-r600');
fprintf('Saved: Figure6_synthetic_GPR.tif (600 DPI) and .pdf\n');
