# SportsBetting
Analysing sports bet pre match start to find any potential positive expected value bets or arbitrage opportunities in miscalculated odds.

The file SBFuncs.py creates the baseline functions that allows SB_playerSpreads.py to:
- access historical player data from nba_api
- find all possible spread bets on player points for upcoming nba games within a certain time frame
- calculate a probability of success for each bet using probability and ML models
- determines if a positive expected value for portfolio size exists based on all middle bets available from the determined spread bets using the Kelly Criterion to determine bet sizes and any opportunities for miscalculated odds

If these outcomes exist (though seldom), then bets can be made with an expectation of gains in the long term even if there are immediate losses.
