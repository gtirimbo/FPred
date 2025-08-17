def expected_goals(mu, mu_home, att_team, def_opponent):
    """
    Calculate expected goals using the log-linear Poisson model.

    Parameters:
    - mu (float): base log-average of away goals
    - mu_home (float): home advantage (log difference)
    - att_team (float): attack rating of the team
    - def_opponent (float): defense rating of the opponent

    Returns:
    - float: expected number of goals
    """
    return np.exp(mu + mu_home + att_team + def_opponent)

from scipy.stats import poisson
import numpy as np

class FootballModel:
    def __init__(self, params):
        self.params = params        # DataFrame with Team as index

    def get_team_stats(self, team):
        return self.params.loc[team]

    def predict(self, team_h, team_a):
        att_h = self.params.loc[team_h, 'att']
        def_h = self.params.loc[team_h, 'def']
        att_a = self.params.loc[team_a, 'att']
        def_a = self.params.loc[team_a, 'def']
        mu = self.params.loc[team_h, 'mu']
        mu_home = self.params.loc[team_h, 'mu_home']

        xG_home = expected_goals(mu, mu_home, att_h, def_a)
        xG_away = expected_goals(mu, 0, att_a, def_h)

        return xG_home, xG_away

    def probability_table(self, team_h, team_a, max_goals=5):
        xG_home, xG_away = self.predict(team_h, team_a)
        home_goals = range(0, max_goals+1)
        away_goals = range(0, max_goals+1)

        table = pd.DataFrame(
            [[poisson.pmf(i, xG_home) * poisson.pmf(j, xG_away)
              for j in away_goals]
             for i in home_goals],
            index=[i for i in home_goals],
            columns=[j for j in away_goals]
        )
        return table

    def outcome_probabilities(self, team_h, team_a, max_goals=5):
        table = self.probability_table(team_h, team_a, max_goals)

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        btts = 0.0
        over_1_5 = 0.0

        for i in table.index:
            for j in table.columns:
                prob = table.loc[i, j]

                # 1X2
                if i > j:
                    p_home += prob
                elif i == j:
                    p_draw += prob
                else:
                    p_away += prob

                # BTTS
                if i > 0 and j > 0:
                    btts += prob

                # Over 1.5 goals
                if (i + j) > 1:
                    over_1_5 += prob

        return {
            "home_team": team_h,
            "away_team": team_a,
            "1": p_home,          # Home win
            "X": p_draw,          # Draw
            "2": p_away,          # Away win
            "1X": p_home + p_draw,
            "12": p_home + p_away,
            "X2": p_draw + p_away,
            "BTTS": btts,
            "Over1.5": over_1_5,
            "Under1.5": 1 - over_1_5
        }
