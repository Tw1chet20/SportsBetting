# SportsBetting

The following five files are for a project that works to calculating market betting implied and local intensities for football over/under total goals based on synthetic historical data of premier league games containg goals data and odds data by intragame time intervals for each game across a season. Synthetic data was generated since the real data is difficult to obtain though the principles of the project maintain. The project uses statistical analysis to determine if the football data matches a poisson distribution, in which case if so this data can be treated like financial derivatives to determine implied and local variable intensities during and before play of the game to be used in a poisson model to price arbitrary over/under bets for the total goals in a game. A basic average goals per game can be used as a multiplier to the determined local intensity to calculate fair prices for specific games. The files are:
- config.py (contains necessary configuration details)
- syntheticFootballDataGen.py (generates synthetic historical football data for multiple seasons then stores as a csv)
- main.py (build classes to analyse the data and apply mathematical and financial techniques to determine the intensities, as well as building a class to price arbitrary and specific bets using a Monte Carlo simulator of a poisson model)
- Control.py (utilisationg of main.py and the data produced by syntheticFootballDataGen.py)
- PeterDivosPhD.py (the rigourous mathematical foundation of which the maths used to determine the intensities is buit from)

The files SBFuncs.py and SB_playerSpreadsEngine.py are used for a project to analyse sports bets pre-match to find any potential positive expected value bets or arbitrage opportunities in miscalculated odds.

The file SBFuncs.py creates the baseline functions that allows SB_playerSpreads.py to:
- access historical player data from nba_api
- find all possible spread bets on player points for upcoming nba games within a certain time frame
- calculate a probability of success for each bet using probability and ML models
- determines if a positive expected value for portfolio size exists based on all middle bets available from the determined spread bets using the Kelly Criterion to determine bet sizes and any opportunities for miscalculated odds

If these outcomes exist (though seldom), then bets can be made with an expectation of gains in the long term even if there are immediate losses.
