# Databricks notebook source

from pyspark.sql.types import StructField, StructType, IntegerType, StringType, BooleanType, DateType, DecimalType
from pyspark.sql.functions import col, when, sum, avg, row_number 
from pyspark.sql.window import Window

# COMMAND ----------

ball_by_ball_df=spark.read.csv("/FileStore/tables/Ball_By_Ball.csv", header=True,inferSchema=True)



# COMMAND ----------

# Display rows where LBW column is True or some specific value
ball_by_ball_df.filter(ball_by_ball_df["lbw"] == True).display(1)


# COMMAND ----------

match_schema=spark.read.csv("/FileStore/tables/Match.csv", header=True,inferSchema=True)

# COMMAND ----------

match_schema.display()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType

# Define the schema for the match dataset
match_schema = StructType([
    StructField("match_sk", IntegerType(), True),
    StructField("match_id", IntegerType(), True),
    StructField("team1", StringType(), True),
    StructField("team2", StringType(), True),
    StructField("match_date", DateType(), True),
    StructField("season_year", IntegerType(), True),
    StructField("venue_name", StringType(), True),
    StructField("city_name", StringType(), True),
    StructField("country_name", StringType(), True),
    StructField("toss_winner", StringType(), True),
    StructField("match_winner", StringType(), True),
    StructField("toss_name", StringType(), True),
    StructField("win_type", StringType(), True),
    StructField("outcome_type", StringType(), True),
    StructField("manofmach", StringType(), True),  # Consider correcting 'manofmach' if it's a typo for 'man_of_match'
    StructField("win_margin", IntegerType(), True),
    StructField("country_id", IntegerType(), True)
])

# Read the CSV file with the provided schema
match_df = spark.read \
    .schema(match_schema) \
    .format("csv") \
    .option("header", "true") \
    .load("/FileStore/tables/Match.csv")  # Corrected file path


# COMMAND ----------

match_df.display(2)

# COMMAND ----------

player_df=spark.read.csv("/FileStore/tables/Player.csv", header=True,inferSchema=True)

# COMMAND ----------

player_schema.display()

# COMMAND ----------


player_match_schema = StructType([
    StructField("player_match_sk", IntegerType(), True),
    StructField("playermatch_key", DecimalType(), True),
    StructField("match_id", IntegerType(), True),
    StructField("player_id", IntegerType(), True),
    StructField("player_name", StringType(), True),
    StructField("dob", DateType(), True),
    StructField("batting_hand", StringType(), True),
    StructField("bowling_skill", StringType(), True),
    StructField("country_name", StringType(), True),
    StructField("role_desc", StringType(), True),
    StructField("player_team", StringType(), True),
    StructField("opposit_team", StringType(), True),
    StructField("season_year", IntegerType(), True),
    StructField("is_manofthematch", BooleanType(), True),
    StructField("age_as_on_match", IntegerType(), True),
    StructField("isplayers_team_won", BooleanType(), True),
    StructField("batting_status", StringType(), True),
    StructField("bowling_status", StringType(), True),
    StructField("player_captain", StringType(), True),
    StructField("opposit_captain", StringType(), True),
    StructField("player_keeper", StringType(), True),
    StructField("opposit_keeper", StringType(), True)
])
# Read the CSV file with the provided schema
player_match_df = spark.read \
    .schema(player_match_schema) \
    .format("csv") \
    .option("header", "true") \
    .load("/FileStore/tables/Player_match-1.csv")  # Corrected file path

# COMMAND ----------

player_match_df.display()

# COMMAND ----------

team_df=spark.read.csv("/FileStore/tables/Team-1.csv", header=True,inferSchema=True)

# COMMAND ----------

team_df.display()

# COMMAND ----------


# Filter to include only valid deliveries (excluding extras like wides and no balls for specific analyses)
ball_by_ball_df = ball_by_ball_df.filter((col("wides") == 0) & (col("noballs")==0))

# Aggregation: Calculate the total and average runs scored in each match and inning
total_and_avg_runs = ball_by_ball_df.groupBy("match_id", "innings_no").agg(
    sum("runs_scored").alias("total_runs"),
    avg("runs_scored").alias("average_runs")
)

# COMMAND ----------

total_and_avg_runs.display()

# COMMAND ----------


# Window Function: Calculate running total of runs in each match for each over
windowSpec = Window.partitionBy("match_id","innings_no").orderBy("over_id")

ball_by_ball_df = ball_by_ball_df.withColumn(
    "running_total_runs",
    sum("runs_scored").over(windowSpec)
)


# COMMAND ----------

ball_by_ball_df.display()

# COMMAND ----------


# Conditional Column: Flag for high impact balls (either a wicket or more than 6 runs including extras)
ball_by_ball_df = ball_by_ball_df.withColumn(
    "high_impact",
    when((col("runs_scored") + col("extra_runs") > 6) | (col("bowler_wicket") == True), True).otherwise(False)
)

# COMMAND ----------

ball_by_ball_df.display()

# COMMAND ----------


from pyspark.sql.functions import year, month, dayofmonth, when

# Extracting year, month, and day from the match date for more detailed time-based analysis
match_df = match_df.withColumn("year", year("match_date"))
match_df = match_df.withColumn("month", month("match_date"))
match_df = match_df.withColumn("day", dayofmonth("match_date"))

# High margin win: categorizing win margins into 'high', 'medium', and 'low'
match_df = match_df.withColumn(
    "win_margin_category",
    when(col("win_margin") >= 100, "High")
    .when((col("win_margin") >= 50) & (col("win_margin") < 100), "Medium")
    .otherwise("Low")
)

# Analyze the impact of the toss: who wins the toss and the match
match_df = match_df.withColumn(
    "toss_match_winner",
    when(col("toss_winner") == col("match_winner"), "Yes").otherwise("No")
)

# Show the enhanced match DataFrame
match_df.display(2)

# COMMAND ----------


from pyspark.sql.functions import lower, regexp_replace

# Normalize and clean player names
player_df = player_df.withColumn("player_name", lower(regexp_replace("player_name", "[^a-zA-Z0-9 ]", "")))

# Handle missing values in 'batting_hand' and 'bowling_skill' with a default 'unknown'
player_df = player_df.na.fill({"batting_hand": "unknown", "bowling_skill": "unknown"})

# Categorizing players based on batting hand
player_df = player_df.withColumn(
    "batting_style",
    when(col("batting_hand").contains("left"), "Left-Handed").otherwise("Right-Handed")
)

# Show the modified player DataFrame
player_df.display(2)

# COMMAND ----------


from pyspark.sql.functions import col, when, current_date, expr

# Add a 'veteran_status' column based on player age
player_match_df = player_match_df.withColumn(
    "veteran_status",
    when(col("age_as_on_match") >= 35, "Veteran").otherwise("Non-Veteran")
)

# Dynamic column to calculate years since debut
player_match_df = player_match_df.withColumn(
    "years_since_debut",
    (year(current_date()) - col("season_year"))
)

# Show the enriched DataFrame
player_match_df.display()

# COMMAND ----------

ball_by_ball_df.createOrReplaceTempView("ball_by_ball")
match_df.createOrReplaceTempView("match")
player_df.createOrReplaceTempView("player")
player_match_df.createOrReplaceTempView("player_match")
team_df.createOrReplaceTempView("team")

# COMMAND ----------


ball_by_ball_df.columns

# COMMAND ----------


top_scoring_batsmen_per_season = spark.sql("""
SELECT 
p.player_name,
m.season_year,
SUM(b.runs_scored) AS total_runs 
FROM ball_by_ball b
JOIN match m ON b.match_id = m.match_id   
JOIN player_match pm ON m.match_id = pm.match_id AND b.striker = pm.player_id     
JOIN player p ON p.player_id = pm.player_id
GROUP BY p.player_name, m.season_year
ORDER BY m.season_year, total_runs DESC
""")

# COMMAND ----------


top_scoring_batsmen_per_season.show(30)

# COMMAND ----------


economical_bowlers_powerplay = spark.sql("""
SELECT 
p.player_name, 
AVG(b.runs_scored) AS avg_runs_per_ball, 
COUNT(b.bowler_wicket) AS total_wickets
FROM ball_by_ball b
JOIN player_match pm ON b.match_id = pm.match_id AND b.bowler = pm.player_id
JOIN player p ON pm.player_id = p.player_id
WHERE b.over_id <= 6
GROUP BY p.player_name
HAVING COUNT(*) >= 1
ORDER BY avg_runs_per_ball, total_wickets DESC
""")
economical_bowlers_powerplay.show()

# COMMAND ----------

toss_impact_individual_matches = spark.sql("""
SELECT m.match_id, m.toss_winner, m.toss_name, m.match_winner,
       CASE WHEN m.toss_winner = m.match_winner THEN 'Won' ELSE 'Lost' END AS match_outcome
FROM match m
WHERE m.toss_name IS NOT NULL
ORDER BY m.match_id
""")
toss_impact_individual_matches.show()


# COMMAND ----------


average_runs_in_wins = spark.sql("""
SELECT p.player_name, AVG(b.runs_scored) AS avg_runs_in_wins, COUNT(*) AS innings_played
FROM ball_by_ball b
JOIN player_match pm ON b.match_id = pm.match_id AND b.striker = pm.player_id
JOIN player p ON pm.player_id = p.player_id
JOIN match m ON pm.match_id = m.match_id
WHERE m.match_winner = pm.player_team
GROUP BY p.player_name
ORDER BY avg_runs_in_wins ASC
""")
average_runs_in_wins.show()

# COMMAND ----------


import matplotlib.pyplot as plt

# COMMAND ----------


# Assuming 'economical_bowlers_powerplay' is already executed and available as a Spark DataFrame
economical_bowlers_pd = economical_bowlers_powerplay.toPandas()

# Visualizing using Matplotlib
plt.figure(figsize=(12, 8))
# Limiting to top 10 for clarity in the plot
top_economical_bowlers = economical_bowlers_pd.nsmallest(10, 'avg_runs_per_ball')
plt.bar(top_economical_bowlers['player_name'], top_economical_bowlers['avg_runs_per_ball'], color='skyblue')
plt.xlabel('Bowler Name')
plt.ylabel('Average Runs per Ball')
plt.title('Most Economical Bowlers in Powerplay Overs (Top 10)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------


import seaborn as sns

# COMMAND ----------



toss_impact_pd = toss_impact_individual_matches.toPandas()

# Creating a countplot to show win/loss after winning toss
plt.figure(figsize=(10, 6))
sns.countplot(x='toss_winner', hue='match_outcome', data=toss_impact_pd)
plt.title('Impact of Winning Toss on Match Outcomes')
plt.xlabel('Toss Winner')
plt.ylabel('Number of Matches')
plt.legend(title='Match Outcome')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------


average_runs_pd = average_runs_in_wins.toPandas()

# Using seaborn to plot average runs in winning matches
plt.figure(figsize=(12, 8))
top_scorers = average_runs_pd.nlargest(10, 'avg_runs_in_wins')
sns.barplot(x='player_name', y='avg_runs_in_wins', data=top_scorers)
plt.title('Average Runs Scored by Batsmen in Winning Matches (Top 10 Scorers)')
plt.xlabel('Player Name')
plt.ylabel('Average Runs in Wins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------


# Execute SQL Query
scores_by_venue = spark.sql("""
SELECT venue_name, AVG(total_runs) AS average_score, MAX(total_runs) AS highest_score
FROM (
    SELECT ball_by_ball.match_id, match.venue_name, SUM(runs_scored) AS total_runs
    FROM ball_by_ball
    JOIN match ON ball_by_ball.match_id = match.match_id
    GROUP BY ball_by_ball.match_id, match.venue_name
)
GROUP BY venue_name
ORDER BY average_score DESC
""")

# COMMAND ----------


# Convert to Pandas DataFrame
scores_by_venue_pd = scores_by_venue.toPandas()

# Plot
plt.figure(figsize=(14, 8))
sns.barplot(x='average_score', y='venue_name', data=scores_by_venue_pd)
plt.title('Distribution of Scores by Venue')
plt.xlabel('Average Score')
plt.ylabel('Venue')
plt.show()

# COMMAND ----------


# Execute SQL Query
dismissal_types = spark.sql("""
SELECT out_type, COUNT(*) AS frequency
FROM ball_by_ball
WHERE out_type IS NOT NULL
GROUP BY out_type
ORDER BY frequency DESC
""")

# COMMAND ----------



# Convert to Pandas DataFrame
dismissal_types_pd = dismissal_types.toPandas()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='frequency', y='out_type', data=dismissal_types_pd, palette='pastel')
plt.title('Most Frequent Dismissal Types')
plt.xlabel('Frequency')
plt.ylabel('Dismissal Type')
plt.show()

# COMMAND ----------


# Execute SQL Query
team_toss_win_performance = spark.sql("""
SELECT team1, COUNT(*) AS matches_played, SUM(CASE WHEN toss_winner = match_winner THEN 1 ELSE 0 END) AS wins_after_toss
FROM match
WHERE toss_winner = team1
GROUP BY team1
ORDER BY wins_after_toss DESC
""")

# COMMAND ----------

team_toss_win_performance.display()

# COMMAND ----------



# Convert to Pandas DataFrame
team_toss_win_pd = team_toss_win_performance.toPandas()

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='wins_after_toss', y='team1', data=team_toss_win_pd)
plt.title('Team Performance After Winning Toss')
plt.xlabel('Wins After Winning Toss')
plt.ylabel('Team')
plt.show()

# COMMAND ----------

# MAGIC %scala
# MAGIC     // Basic arithmetic operations
# MAGIC     val a = 10
# MAGIC     val b = 5
# MAGIC
# MAGIC     val addition = a + b
# MAGIC     val subtraction = a - b
# MAGIC     val multiplication = a * b
# MAGIC     val division = a / b
# MAGIC     val remainder = a % b
# MAGIC
# MAGIC     // Display results
# MAGIC     println(s"Addition: $a + $b = $addition")
# MAGIC     println(s"Subtraction: $a - $b = $subtraction")
# MAGIC     println(s"Multiplication: $a * $b = $multiplication")
# MAGIC     println(s"Division: $a / $b = $division")
# MAGIC     println(s"Remainder: $a % $b = $remainder")
# MAGIC
# MAGIC     // More mathematical operations
# MAGIC     val number = 16.0
# MAGIC     val squareRoot = Math.sqrt(number)
# MAGIC     val power = Math.pow(number, 2)
# MAGIC
# MAGIC     // Display results
# MAGIC     println(s"Square Root of $number = $squareRoot")
# MAGIC     println(s"$number to the power of 2 = $power")
# MAGIC   
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %r
# MAGIC # Load the built-in 'mtcars' dataset
# MAGIC data(mtcars)
# MAGIC
# MAGIC # Save the plot as a PNG image without displaying it
# MAGIC png(file = "hp_vs_mpg.png", width = 800, height = 600)
# MAGIC
# MAGIC # Create a scatter plot: Horsepower (hp) vs. Miles per Gallon (mpg)
# MAGIC plot(mtcars$hp, mtcars$mpg,
# MAGIC      xlab = "Horsepower (hp)",
# MAGIC      ylab = "Miles per Gallon (mpg)",
# MAGIC      main = "Horsepower vs MPG",
# MAGIC      col = "purple",
# MAGIC      pch = 19,
# MAGIC      cex = 1.5)  # Adjust point size as needed
# MAGIC
# MAGIC # Add a regression line using 'abline'
# MAGIC model <- lm(mpg ~ hp, data = mtcars)
# MAGIC abline(model, col = "red")
# MAGIC
# MAGIC # Add a text label for the regression equation
# MAGIC equation <- paste("y =", round(coef(model)[1], 2), "+", round(coef(model)[2], 2), "x")
# MAGIC text(x = 300, y = 25, labels = equation, col = "red")
# MAGIC
# MAGIC # Close the PNG device to finalize and save the plot
# MAGIC dev.off()

# COMMAND ----------

# MAGIC %r
# MAGIC # Load the built-in 'mtcars' dataset
# MAGIC data(mtcars)
# MAGIC
# MAGIC # Define the path using DBFS
# MAGIC file_path <- "/dbfs/tmp/hp_vs_mpg.png"
# MAGIC
# MAGIC # Create the directory if it does not exist
# MAGIC dir.create("/dbfs/tmp", showWarnings = FALSE)
# MAGIC
# MAGIC # Save the plot as a PNG image
# MAGIC png(file = file_path, width = 800, height = 600)
# MAGIC
# MAGIC # Create a scatter plot: Horsepower (hp) vs. Miles per Gallon (mpg)
# MAGIC plot(mtcars$hp, mtcars$mpg,
# MAGIC      xlab = "Horsepower (hp)",
# MAGIC      ylab = "Miles per Gallon (mpg)",
# MAGIC      main = "Horsepower vs MPG",
# MAGIC      col = "purple",
# MAGIC      pch = 19,
# MAGIC      cex = 1.5)  # Adjust point size as needed
# MAGIC
# MAGIC # Add a regression line using 'abline'
# MAGIC model <- lm(mpg ~ hp, data = mtcars)
# MAGIC abline(model, col = "red")
# MAGIC
# MAGIC # Add a text label for the regression equation
# MAGIC equation <- paste("y =", round(coef(model)[1], 2), "+", round(coef(model)[2], 2), "x")
# MAGIC text(x = 300, y = 25, labels = equation, col = "red")
# MAGIC
# MAGIC # Close the PNG device to finalize and save the plot
# MAGIC dev.off()
# MAGIC

# COMMAND ----------

# MAGIC %r
# MAGIC # Load the built-in 'mtcars' dataset
# MAGIC data(mtcars)
# MAGIC
# MAGIC # Define the path using DBFS
# MAGIC file_path <- "/dbfs/Users/mechinelearningai@gmail.com/scala course/Data Engineering Project Pysparks/hp_vs_mpg.png"
# MAGIC
# MAGIC # Create the directory if it does not exist
# MAGIC dir.create("/dbfs/Users/mechinelearningai@gmail.com/scala course/Data Engineering Project Pysparks", showWarnings = FALSE, recursive = TRUE)
# MAGIC
# MAGIC # Save the plot as a PNG image
# MAGIC png(file = file_path, width = 800, height = 600)
# MAGIC
# MAGIC # Create a scatter plot: Horsepower (hp) vs. Miles per Gallon (mpg)
# MAGIC plot(mtcars$hp, mtcars$mpg,
# MAGIC      xlab = "Horsepower (hp)",
# MAGIC      ylab = "Miles per Gallon (mpg)",
# MAGIC      main = "Horsepower vs MPG",
# MAGIC      col = "purple",
# MAGIC      pch = 19,
# MAGIC      cex = 1.5)  # Adjust point size as needed
# MAGIC
# MAGIC # Add a regression line using 'abline'
# MAGIC model <- lm(mpg ~ hp, data = mtcars)
# MAGIC abline(model, col = "red")
# MAGIC
# MAGIC # Add a text label for the regression equation
# MAGIC equation <- paste("y =", round(coef(model)[1], 2), "+", round(coef(model)[2], 2), "x")
# MAGIC text(x = 300, y = 25, labels = equation, col = "red")
# MAGIC
# MAGIC # Close the PNG device to finalize and save the plot
# MAGIC dev.off()
# MAGIC

# COMMAND ----------


