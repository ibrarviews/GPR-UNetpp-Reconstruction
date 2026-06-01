function FK = compute_fk(D)
% Compute F-K amplitude spectrum.
% - fftshift on wavenumber axis only (columns)
% - positive frequencies only (rows 1 to Nt/2)
% - returns amplitude (not power)
    Nt   = size(D, 1);
    S    = fft2(D);
    S    = fftshift(S, 2);              % shift kx axis only
    FK   = abs(S(1:floor(Nt/2), :));   % positive freq half
end
