# Flux loss weight sensitivity summary

All runs use the same random seed, architecture, collocation sampling strategy, and optimizer settings.

- Best RMSE against finite-volume reference: lambda=10, RMSE=2.292372e-04
- Best post-startup surface-flux RMS error: lambda=100, error=3.851465e-04
- Best post-startup surface-flux max error: lambda=20, error=1.772737e-03
- Selected default for the paper calculation: lambda=20. This is not the single best value for every metric, but it gives a stronger balance than lambda=40: lower RMSE, lower mass-balance error, and lower post-startup surface-flux maximum error while keeping the PDE residual at the same order of magnitude.
- Use `summary.csv` for the complete numerical table.
