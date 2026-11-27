# Quantitative findings

## Equity and commodity

- The BS control log-log standard-deviation slope was 0.476, against the complete-market small-step benchmark of 0.5.
- The Bates vanilla variance-fit intercepts were 0.6778 for sticky-strike delta and 0.1018 for BS delta-vega.
- The largest FFT versus COS difference in the benchmark grid was 0.0003 implied-volatility bp.
- First- and second-order Taylor residual RMSE were 69.8% and 11.0% of full BS P&L RMSE.
- Daily full recalibration fell back on 21.1% of base vanilla path-days.

## Rates

- Both OIS curves reprice the complete input strip within 1.7e-11 bp.
- The fine base G2++ Bermudan ADI mark is 395.10 notional bp. Grid refinement changes it by 0.36 bp.
- The base paired HW exercise loss averages 2.33 notional bp with a 95% interval [-0.96, 5.62].
- Against true G2++ marks, the HW desk PLA classified 0 green, 0 amber and 16 red paths, with 1 actual exercise in the windows.
- With 1-day rebalancing in the base G2++ world, the three-swap key-rate hedge has a discounted P&L SD of 22.4 bp, against 117.2 bp for one-swap delta and 8.9 bp for the full-information swap control.
- Across both G2++ worlds and 1, 5, 21-day rebalancing, the key-rate and vega hedge SD stays between 17.1 and 31.6 bp.
- On the shared quote snapshot against the front-office G2++, the HW risk desk has 0 green, 0 amber and 16 red paths.
- On the shared quote snapshot against the front-office G2++, the G2++ risk desk has 16 green, 0 amber and 0 red paths.
- In the G2++ PLA sensitivities, refitting rho daily on both desks gives 16 green, 0 amber and 0 red; fitting the risk desk to ATM quotes only gives 1 green, 8 amber and 7 red paths.
