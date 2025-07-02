# Import Regular Expression and Matplotlib libraries
import re
import matplotlib.pyplot as plt

# Initialize an empty list to hold the results
results = []

# Open the 'results.txt' file for reading
with open('results.txt', 'r') as f:
    # Open the 'results.txt' file for reading
    for line in f:
        # Use regular expressions to search for lines that match the pattern 'Round [number], Test Acc: [number]'
        # The regular expression r'Round (\d+), Test Acc: (\d+\.\d+)' captures:
        # (\d+) - One or more digits that represent the round number
        # (\d+\.\d+) - A floating-point number that represents the test accuracy
        match = re.search(r'Round (\d+), Test Acc: (\d+\.\d+)', line)

        # If the line matches the pattern
        if match:
            # Extract the round number and test accuracy and append them as a tuple to the 'results' list
            # match.group(1) will contain the round number, match.group(2) will contain the test accuracy
            results.append((int(match.group(1)), float(match.group(2))))

# Unzip the list of tuples into two lists: 'rounds' and 'accuracies'
# rounds will contain the round numbers, and accuracies will contain the test accuracies
rounds, accuracies = zip(*results)

# Print the extracted accuracies for debugging purposes
print(accuracies)

# Create a plot
# X-axis will have the round numbers
# Y-axis will have the test accuracies
# Each point in the plot will be marked with 'o'
plt.plot(rounds, accuracies, marker='o')

# Label the X-axis as 'Round'
plt.xlabel('Round')

# Label the Y-axis as 'Test Accuracy'
plt.ylabel('Test Accuracy')

# Display the plot
plt.show()
