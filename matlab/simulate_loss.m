function [epochs, train_loss, val_loss] = simulate_loss(n_epochs, init_loss, final_loss)
% SIMULATE_LOSS  Generate a realistic U-Net++ training/validation loss curve.
%
%   [epochs, train_loss, val_loss] = simulate_loss(n_epochs, init_loss, final_loss)
%
%   n_epochs   : total number of training epochs (e.g. 100)
%   init_loss  : starting loss value (e.g. 0.520)
%   final_loss : converged loss value (e.g. 0.031)
%
%   Returns:
%   epochs     : 1 x n_epochs array of epoch indices
%   train_loss : 1 x n_epochs training loss
%   val_loss   : 1 x n_epochs validation loss

    epochs = 1 : n_epochs;
    t      = (epochs - 1) / (n_epochs - 1);   % normalised 0→1

    % Cosine decay from init_loss to final_loss
    cos_decay = final_loss + 0.5*(init_loss - final_loss) * (1 + cos(pi*t));

    % Warm-up: slow start for first 5 epochs
    warmup_epochs = 5;
    warmup_factor = ones(1, n_epochs);
    for i = 1:warmup_epochs
        warmup_factor(i) = 0.4 + 0.6*(i/warmup_epochs);
    end
    cos_decay(1:warmup_epochs) = ...
        init_loss*(1 - warmup_factor(1:warmup_epochs)) + ...
        cos_decay(1:warmup_epochs).*warmup_factor(1:warmup_epochs);

    % Stochastic noise — larger early, smaller late
    noise_scale    = 0.006;
    noise_envelope = noise_scale*exp(-3.5*t) + noise_scale*0.25;
    noise          = randn(1, n_epochs) .* noise_envelope;

    train_loss = max(cos_decay + noise, 0);

    % Validation loss: slightly above training, same trend
    val_offset = 0.008 + 0.015*exp(-4.0*t);
    val_noise  = randn(1, n_epochs) * noise_scale * 0.9;
    val_loss   = max(train_loss + val_offset + val_noise, 0);
end
