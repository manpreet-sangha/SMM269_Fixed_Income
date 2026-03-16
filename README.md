# SMM269 Fixed Income — Pricing a Structured Product

Coursework for **SMM269 Fixed Income** (Term 2, City St George's, University of London).

## Bond

**UniCredit S.p.A. Variable Rate Bond 2024–2034** (ISIN: IT0005599110)

- Quarterly coupon linked to 3-month EURIBOR with a 1.60× participation rate
- Embedded collar: floor at 0 %, cap at 5.45 %
- Coupon formula: `c(r) = max(0 %, min(1.60 × 3M EURIBOR, 5.45 %))`
- Maturity: 12 June 2034, 100 % redemption at par

## Repository layout

```
docs/                        Reference documents and Bloomberg screenshots
input_data/                  Raw market data inputs
misc/                        Auxiliary scripts, LaTeX source, bond data
Pricing_Structured_Product/  All pricing and analysis code
├── common/                  Shared data (EURIBOR CSV, bond details)
├── run_all.py               Master runner — executes every question script
├── Q1/                      Bond structure, coupon profile & QuantLib replication
├── Q3/                      EUR term-structure bootstrapping (deposits + IRS)
├── Q4/                      Scenario analysis (best / worst / base case)
├── Q5/                      Bachelier caplet/floorlet calibration
├── Q6/                      Credit Valuation Adjustment (CVA)
├── Q8/                      Fair-value assembly
├── Q9/                      Multi-panel cash-flow chart
├── Q10/                     Market comparison & Z-spread
├── Q11/                     PCA on EUR IRS par-rate changes
├── Q12/                     PCA scenario analysis & repricing
├── Q13/                     Hedging strategy (2Y + 10Y receiver swaps)
├── Q14/                     Hedge effectiveness under PCA scenarios
├── Q15/                     CDS spread sensitivity (CS-DV01)
├── Q16/                     CDS hedge sizing
├── Q17/                     Factor model for price changes
├── Q18/                     Monte-Carlo VaR & Expected Shortfall
├── Q19/                     Analytical VaR / ES vs Monte Carlo
├── Q20/                     Marginal & Component VaR decomposition
├── Q21/                     Hedged-portfolio VaR decomposition
└── Q22/                     Summary consolidation (CSV for LaTeX)
```

## Quick start

```bash
pip install QuantLib numpy scipy matplotlib pandas
```

Run all questions at once:

```bash
cd Pricing_Structured_Product
python run_all.py
```

Or run a single question:

```bash
cd Pricing_Structured_Product/Q5
python q5_main.py
```

Each `Q*/` folder writes its outputs (figures, CSVs, text) into a local `output/` sub-folder.

## Dependencies

| Package    | Purpose                              |
|------------|--------------------------------------|
| QuantLib   | Schedule generation, option pricing  |
| NumPy      | Numerical computation                |
| SciPy      | Optimisation, statistics             |
| Matplotlib | Charts and table images              |
| pandas     | Data handling and CSV I/O            |

Python 3.12+ recommended.
