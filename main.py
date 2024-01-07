from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder.appName("PythonLineCounter").getOrCreate()

# Load the text file into a DataFrame
text_file = spark.read.text("file.txt")

# Filter lines that start with 'Python'
lines_starting_with_python = text_file.filter(text_file.value.startswith("Python"))

# Count the lines
count = lines_starting_with_python.count()

# Print the count
print(f"Number of lines starting with 'Python': {count}")

# Stop the Spark session
spark.stop()
