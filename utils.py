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
