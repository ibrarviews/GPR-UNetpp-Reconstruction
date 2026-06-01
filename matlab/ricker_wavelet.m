function wav = ricker_wavelet(freq_mhz, dt_ns, ~)
% RICKER_WAVELET  Generate a compact Ricker wavelet (3 dominant periods).
%   The third argument (t_max_ns) is ignored — wavelet duration is always
%   set to 3 periods to keep it compact and avoid filling the whole trace.
%
%   wav = ricker_wavelet(freq_mhz, dt_ns)

    f       = freq_mhz * 1e6;              % Hz
    T_period = 1.0 / f * 1e9;              % period in ns
    dur_ns  = 3.0 * T_period;              % 3 periods total
    t       = (0 : dt_ns : dur_ns) * 1e-9; % seconds
    t0      = 1.0 / f;                     % one-period delay
    u       = pi * f * (t - t0);
    wav     = (1.0 - 2.0*u.^2) .* exp(-u.^2);
    wav     = wav / max(abs(wav));
    wav     = wav(:);
end
