# PySpark Line Counter

## Overview

This PySpark project is a simple yet illustrative example showcasing the use of Apache Spark to count the number of
lines in a text file that start with the word 'Python'. The primary aim is to demonstrate the distributed processing
capabilities of Spark for handling large-scale data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)

## Prerequisites

Before running the project, ensure the following prerequisites are met:

- [Apache Spark](https://spark.apache.org/downloads.html) is installed.
- Python 3.11 is installed.
- Java is installed (required for Spark).

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/python-line-counter.git
   cd python-line-counter
   ```

2. **Install Dependencies:**

   ```bash
   pip install pyspark
   ```

3. **Set Up Spark Environment:**

    - Extract the downloaded Spark archive to a directory.

    - Set the `SPARK_HOME` environment variable:

      ```bash
      export SPARK_HOME=/path/to/your/spark/directory
      ```

    - Add Spark's `bin` directory to your `PATH`:

      ```bash
      export PATH=$SPARK_HOME/bin:$PATH
      ```

4. **Run the PySpark Script:**

   ```bash
   spark-submit python_line_counter.py
   ```

## Project Structure

- **`main.py`**: Main Python line counter PySpark script for line counting.
- **`file.txt`**: Sample input text file (replace it with your own file).

## Results

The script processes the provided text file and outputs the number of lines starting with 'Python'.

