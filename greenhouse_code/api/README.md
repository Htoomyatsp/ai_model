## Endpoint
POST /predict

## Request Format
- Content-Type: application/json
- Shape: 20 timesteps × 21 features
- Format: A list of 20 lists (each inner list contains 21 float values)
See the example_request1.json and example_request2.json for request examples

Each inner list must follow this exact order of features per timestep

- 0	CO2air
- 1	Cum_irr
- 2	Tair
- 3	Tot_PAR
- 4	Ventwind
- 5	AssimLight
- 6	VentLee
- 7	HumDef
- 8	co2_dos
- 9	PipeGrow
- 10 EnScr
- 11 BlackScr
- 12 Windsp
- 13 Winddir
- 14 Tout
- 15 Rhout
- 16 AbsHumOut
- 17 PARout
- 18 Iglob
- 19 Pyrgeo
- 20 RadSum

## Response Format
```
{
  "prediction": [
    410.2, 31.5, 22.9, 710.3, 0.7, 82.5, 0.35, 5.3, 1.7, 31.0,
    0.25, 0.15, 1.4, 180.5, 17.9, 61.0, 8.3, 505.0, 905.0, 355.0, 755.0
  ]
}
```
- 0	CO2air
- 1	Cum_irr
- 2	Tair
- 3	Tot_PAR
- 4	Ventwind
- 5	AssimLight
- 6	VentLee
- 7	HumDef
- 8	co2_dos
- 9	PipeGrow
- 10 EnScr
- 11 BlackScr
- 12 Windsp
- 13 Winddir
- 14 Tout
- 15 Rhout
- 16 AbsHumOut
- 17 PARout
- 18 Iglob
- 19 Pyrgeo
- 20 RadSum

## Validation
- Input must be exactly 20 timesteps
- Each timestep must include exactly 21 values in the order mentioned previously
- Values should be raw/unscaled, the API applies internal normalization