# Script 04 plotting summary - see folder (04_CromTobitAnalysis)
Script 04 is the core trend-analysis step in the workflow
- preproccesess chemistry and water-level data
- assign trend periods (TERM) using cutoffs, special rules/overrides
- model on: time only or time + river stage (with lag)
A result may be:
- a single trend summary object, or
- a list of trend summary objects when multiple TERM periods exist

# Script 05 plotting summary - see folder (05_ReportFigures)
Note: all R code functions/helpers used in <mark>05_ReportFigures.R</mark> saved in <mark>./r_function_extract</mark><br>
Script 05 is the reporting layer for the Tobit workflow.<br>
Uses trend-analysis data from Script 04, water-level trend data, well metadata, and GIS base data, then produces:
- one PDF report per OU
- one page per well with summary table, map and plots

## Main workflow
1. Load trend and water-level data
2. Load well metadata, GIS base layers
3. Group wells by OU
4. For each well, extract:
   - analyte name and units
   - well coordinates / OU
   - whether river stage is used
   - associated water-level object
   - lag(s) / regression info for each TERM
5. Choose one of two page templates:
   - `pltReport(...)` if trend / lag output exists
   - `pltPlaceholder(...)` if no regression analysis was performed

## Page layout 
### Full report page (`pltReport`)
The page contains:
- header
- header summary table
- location map
- legend
- top time-series panel: observed water level + river stage
- middle panel: observed chromium concentrations + river stage
- lower-middle panel: observed + model-predicted concentrations
- bottom panel: regression summary / significance display

### Placeholder page (`pltPlaceholder`) - if no trend reported
Uses the same upper layout:
- header
- header summary table
- location map
- legend
- optional water-level panel
- chromium observations panel

The bottom plot area is replaced with a simple **No Regression Analysis Performed** message.

## Notes
- Chromium observations are plotted on a log scale
- Non-detects are highlighted separately (red triangle)
- Each trend TERM is shown with its own colour / symbol styling
- If water-level data exist, the top panel also shows the screened interval
- River stage is omitted for wells listed in `NoRS`
- Script 05 creates a report summary table for each well using `tblSummary(...)`


