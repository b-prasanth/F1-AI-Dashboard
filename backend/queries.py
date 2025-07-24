import pandas as pd

def get_calendar_query():
    query = """
    SELECT 
        c.round,
        c.GrandPrix as grand_prix,
        c.circuit,
        c.date,
        q.driver as PoleSitter,
        q.driver_id as PoleSitterId,
        r.driver as Winner,
        r.driver_id as WinnerId,
        c.status
    FROM 
        f1_calendar c
    LEFT JOIN 
        qualifying_results q ON c.year = q.year AND c.round = q.round AND q.position = 1
    LEFT JOIN 
        race_results r ON c.year = r.year AND c.round = r.round AND r.finish_position = 1
    WHERE 
        c.year = :selected_year
    """
    return query

def get_available_years_query():
    query = "SELECT DISTINCT year FROM drivers_championship;"
    return query

def get_constructors_championship_query():
    query = "SELECT position AS Position, constructor AS Team, points AS Points, wins AS Wins, constructor_id FROM constructors_championship WHERE year = :year;"
    return query

def get_drivers_championship_query():
    query = """
    WITH driver_teams AS (
        SELECT 
            driver,
            driver_id,
            SUM(points) as total_points,
            SUM(wins) as total_wins,
            GROUP_CONCAT(DISTINCT constructors ORDER BY id SEPARATOR '/') as all_teams
        FROM drivers_championship
        WHERE year = :year
        GROUP BY driver, driver_id
    ),
    final_standings AS (
        SELECT 
            driver,
            driver_id,
            all_teams as team,
            total_points as points,
            total_wins as wins,
            ROW_NUMBER() OVER (ORDER BY total_points DESC, total_wins DESC, driver) as position
        FROM driver_teams
    )
    SELECT 
        position AS Position,
        driver AS Driver,
        driver_id,
        team AS Team,
        points AS Points,
        wins AS Wins
    FROM final_standings
    ORDER BY position;
    """
    return query

def get_race_results_by_gp():
    query = """
    SELECT 
        round AS Round,
        grand_prix AS GrandPrix,
        driver AS Driver,
        finish_position AS FinishPosition,
        grid_position AS GridPosition
    FROM 
        race_results
    WHERE 
        year = :year AND 
        grand_prix = :grand_prix
    ORDER BY 
        finish_position ASC
    """
    return query

def get_race_results_query():
    query = """
    SELECT 
        year, 
        round, 
        grand_prix, 
        driver, 
        driver_id, 
        constructor, 
        finish_position
    FROM 
        race_results
    WHERE 
        year = :year AND 
        driver_id IN (
            SELECT DISTINCT driver_id 
            FROM drivers_championship 
            WHERE year = :year
        )
    """
    return query

def get_qualifying_results_query():
    query = """
    SELECT 
        year, 
        round, 
        grand_prix, 
        driver, 
        driver_id, 
        constructor, 
        position
    FROM 
        qualifying_results
    WHERE 
        year = :year AND 
        driver_id IN (
            SELECT DISTINCT driver_id 
            FROM drivers_championship 
            WHERE year = :year
        )
    """
    return query

def get_winning_drivers():
    query = "SELECT driver, wins FROM drivers_championship WHERE year = :year AND wins>=1;"
    return query

def get_winning_constructors():
    query = "SELECT constructor, wins FROM constructors_championship WHERE year = :year AND wins>=1"
    return query

def get_race_query():
    race_query = """
    SELECT year, round, constructor, points
    FROM race_results
    WHERE year = :year
    ORDER BY round
    """
    return race_query

def get_sprint_race_query():
    sprint_query = """
    SELECT year, round, constructor, points
    FROM sprint_race_results
    WHERE year = :year
    ORDER BY round
    """
    return sprint_query

def get_points_history_query():
    query = """
    SELECT year, round, driver, constructor, points
    FROM race_results
    WHERE year = :year
    UNION ALL
    SELECT year, round, driver, constructor, points
    FROM sprint_race_results
    WHERE year = :year
    ORDER BY round
    """
    return query

def get_team_wins_season():
    query = """SELECT constructor, wins FROM constructors_championship WHERE year = :year AND wins>=1;"""
    return query

def get_driver_wins_season():
    query = """SELECT driver, wins FROM drivers_championship WHERE year = :year AND wins>=1;"""
    return query

def get_driver_poles_season():
    query = """SELECT COUNT(*) as poles, driver FROM race_results WHERE year = :year AND grid_position = 1 GROUP BY driver; """
    return query

def get_team_poles_season():
    query = """SELECT COUNT(*) as poles, constructor FROM race_results WHERE year = :year AND grid_position = 1 GROUP BY constructor; """
    return query

def get_team_drivers():
    query = """
    SELECT DISTINCT constructors as Team, Driver
    FROM drivers_championship
    WHERE year = :year
    ORDER BY constructors, Driver
    """
    return query

def get_quali_perf_data():
    query = """
    WITH constructor_quali_performance AS (
        SELECT 
            year,
            constructor,
            COUNT(*) as total_entries,
            AVG(CAST(position AS DECIMAL(10,2))) as avg_quali_position,
            MIN(position) as best_quali,
            MAX(position) as worst_quali,
            STDDEV(CAST(position AS DECIMAL(10,2))) as quali_consistency,
            COUNT(CASE WHEN position <= 3 THEN 1 END) as front_row_lockouts,
            COUNT(CASE WHEN position <= 10 THEN 1 END) as q3_appearances
        FROM qualifying_results
        WHERE year = :year 
            AND position IS NOT NULL
            AND NOT (driver = 'Liam Lawson' AND constructor = 'Red Bull Racing')
            AND NOT (driver = 'Yuki Tsunoda' AND constructor = 'Racing Bulls')
        GROUP BY year, constructor
    )
    SELECT 
        year,
        constructor,
        total_entries,
        ROUND(avg_quali_position, 2) as avg_qualifying_position,
        best_quali as best_grid_position,
        ROUND(quali_consistency, 2) as qualifying_consistency,
        front_row_lockouts,
        q3_appearances,
        ROUND((q3_appearances * 100.0 / total_entries), 1) as q3_percentage,
        ROUND(100 - avg_quali_position + (q3_appearances * 2), 2) as car_quali_index
    FROM constructor_quali_performance
    ORDER BY year, car_quali_index DESC;
    """
    return query

def get_race_perf_data():
    query = """
    WITH constructor_race_performance AS (
        SELECT 
            year,
            constructor,
            COUNT(*) as total_races,
            AVG(CASE 
                WHEN finish_position IS NOT NULL 
                THEN CAST(finish_position AS DECIMAL(10,2)) 
                END) as avg_finish_position,
            COUNT(CASE WHEN finish_position = 1 THEN 1 END) as wins,
            COUNT(CASE WHEN finish_position <= 3 THEN 1 END) as podiums,
            COUNT(CASE WHEN finish_position <= 10 THEN 1 END) as points_finishes,
            COUNT(CASE WHEN status = 'Finished' THEN 1 END) as completed_races,
            SUM(COALESCE(points, 0)) as total_points
        FROM race_results
        WHERE year = :year
            AND NOT (driver = 'Liam Lawson' AND constructor = 'Red Bull Racing')
            AND NOT (driver = 'Yuki Tsunoda' AND constructor = 'Racing Bulls')
        GROUP BY year, constructor
    )
    SELECT 
        year,
        constructor,
        total_races,
        ROUND(avg_finish_position, 2) as avg_finish_position,
        wins,
        podiums,
        points_finishes,
        ROUND((completed_races * 100.0 / total_races), 1) as reliability_rate,
        total_points,
        ROUND((wins * 25) + (podiums * 10) + (points_finishes * 3) + 
              (completed_races * 100.0 / total_races) + 
              (total_points / 10), 2) as car_race_index
    FROM constructor_race_performance
    ORDER BY year, car_race_index DESC;
    """
    return query

def get_driver_quali_perf():
    query = """
    WITH teammate_pairs AS (
        SELECT 
            q1.year,
            q1.round,
            q1.driver as driver1,
            q1.driver_id as driver1_id,
            q1.position as driver1_quali_pos,
            q2.driver as driver2,
            q2.driver_id as driver2_id,
            q2.position as driver2_quali_pos,
            q1.constructor
        FROM qualifying_results q1
        JOIN qualifying_results q2 ON 
            q1.year = q2.year 
            AND q1.round = q2.round 
            AND q1.constructor = q2.constructor 
            AND q1.driver_id < q2.driver_id
        WHERE q1.year = :year
            AND q1.position IS NOT NULL 
            AND q2.position IS NOT NULL
            AND NOT (q1.driver = 'Liam Lawson' AND q1.constructor = 'Red Bull Racing')
            AND NOT (q2.driver = 'Liam Lawson' AND q2.constructor = 'Red Bull Racing')
            AND NOT (q1.driver = 'Yuki Tsunoda' AND q1.constructor = 'Racing Bulls')
            AND NOT (q2.driver = 'Yuki Tsunoda' AND q2.constructor = 'Racing Bulls')
    ),
    driver_teammate_performance AS (
        SELECT 
            year,
            driver1 as driver,
            driver1_id as driver_id,
            constructor,
            COUNT(*) as races_compared,
            AVG(CAST(driver1_quali_pos - driver2_quali_pos AS DECIMAL(10,2))) as avg_quali_gap,
            STDDEV(CAST(driver1_quali_pos - driver2_quali_pos AS DECIMAL(10,2))) as quali_consistency,
            SUM(CASE WHEN driver1_quali_pos < driver2_quali_pos THEN 1 ELSE 0 END) as times_ahead
        FROM teammate_pairs
        GROUP BY year, driver1, driver1_id, constructor
        UNION ALL
        SELECT 
            year,
            driver2 as driver,
            driver2_id as driver_id,
            constructor,
            COUNT(*) as races_compared,
            AVG(CAST(driver2_quali_pos - driver1_quali_pos AS DECIMAL(10,2))) as avg_quali_gap,
            STDDEV(CAST(driver2_quali_pos - driver1_quali_pos AS DECIMAL(10,2))) as quali_consistency,
            SUM(CASE WHEN driver2_quali_pos < driver1_quali_pos THEN 1 ELSE 0 END) as times_ahead
        FROM teammate_pairs
        GROUP BY year, driver2, driver2_id, constructor
    )
    SELECT 
        year,
        driver,
        driver_id,
        constructor,
        races_compared,
        ROUND(avg_quali_gap, 2) as avg_qualifying_gap_vs_teammate,
        ROUND(quali_consistency, 2) as qualifying_consistency,
        times_ahead,
        ROUND((times_ahead * 100.0 / races_compared), 1) as percent_ahead_of_teammate,
        ROUND(100 - (avg_quali_gap + COALESCE(quali_consistency, 0)), 2) as driver_quali_index
    FROM driver_teammate_performance
    ORDER BY year, driver_quali_index DESC;
    """
    return query

def get_driver_race_perf():
    query = """
    WITH race_performance AS (
        SELECT 
            year,
            driver,
            driver_id,
            constructor,
            COUNT(*) as total_races,
            AVG(CASE 
                WHEN grid_position IS NOT NULL AND finish_position IS NOT NULL 
                THEN CAST(grid_position - finish_position AS DECIMAL(10,2))
                END) as avg_grid_gain,
            COUNT(CASE WHEN finish_position <= 10 THEN 1 END) as points_finishes,
            COUNT(CASE WHEN finish_position = 1 THEN 1 END) as wins,
            COUNT(CASE WHEN status = 'Finished' THEN 1 END) as completed_races
        FROM race_results
        WHERE year = :year 
            AND grid_position IS NOT NULL
            AND NOT (driver = 'Liam Lawson' AND constructor = 'Red Bull Racing')
            AND NOT (driver = 'Yuki Tsunoda' AND constructor = 'Racing Bulls')
        GROUP BY year, driver, driver_id, constructor
    )
    SELECT 
        year,
        driver,
        driver_id,
        constructor,
        total_races,
        ROUND(avg_grid_gain, 2) as avg_positions_gained,
        points_finishes,
        wins,
        ROUND((completed_races * 100.0 / total_races), 1) as reliability_rate,
        ROUND((avg_grid_gain * 10) + (points_finishes * 2) + (wins * 5) + 
              (completed_races * 100.0 / total_races), 2) as driver_race_index
    FROM race_performance
    ORDER BY year, driver_race_index DESC;
    """
    return query

def get_driver_summary():
    query = """
    WITH driver_summary AS (
        SELECT 
            dc.year,
            dc.driver,
            dc.driver_id,
            dc.position as championship_position,
            dc.points as championship_points,
            dc.constructors as team,
            AVG(CAST(qr.position AS DECIMAL(10,2))) as avg_quali_pos,
            AVG(CASE WHEN rr.finish_position IS NOT NULL 
                THEN CAST(rr.finish_position AS DECIMAL(10,2)) END) as avg_race_pos
        FROM drivers_championship dc
        LEFT JOIN qualifying_results qr ON dc.year = qr.year AND dc.driver_id = qr.driver_id
        LEFT JOIN race_results rr ON dc.year = rr.year AND dc.driver_id = rr.driver_id
        WHERE dc.year = :year
            AND NOT (dc.driver = 'Liam Lawson' AND dc.constructors = 'Red Bull Racing')
            AND NOT (dc.driver = 'Yuki Tsunoda' AND dc.constructors = 'Racing Bulls')
        GROUP BY dc.year, dc.driver, dc.driver_id, dc.position, dc.points, dc.constructors
    )
    SELECT 
        year,
        driver,
        championship_position,
        championship_points,
        team,
        ROUND(avg_quali_pos, 2) as avg_qualifying_position,
        ROUND(avg_race_pos, 2) as avg_race_position,
        ROUND((championship_points / 10) + 
              (25 - avg_quali_pos) + 
              (25 - avg_race_pos), 2) as overall_driver_index
    FROM driver_summary
    WHERE avg_quali_pos IS NOT NULL
    ORDER BY year, overall_driver_index DESC;
    """
    return query

def get_gp_race_results_query():
    query = """
    SELECT year, round, grand_prix, circuit, driver, constructor, finish_position, grid_position
    FROM race_results
    WHERE grand_prix = :grand_prix
    AND year = :year ORDER BY finish_position
    """
    return query

def get_gp_qualifying_results_query():
    query = """
    SELECT year, round, grand_prix, driver, constructor, position
    FROM qualifying_results
    WHERE grand_prix = :grandprix
    AND year = :year ORDER BY position
    """
    return query

def get_driver_podiums(year):
    query = """SELECT driver AS Driver, COUNT(*) as Podiums FROM race_results WHERE year = :year AND finish_position <=3 GROUP BY Driver ORDER BY Podiums DESC;"""
    return query

def get_constructor_podiums(year):
    query = """SELECT constructor AS Constructor, COUNT(*) as Wins FROM race_results WHERE year = :year AND finish_position <= 3 GROUP BY Constructor ORDER BY Wins DESC;"""
    return query