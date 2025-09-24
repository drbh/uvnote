# Python Basics

## Lists and Loops

```python
# Working with lists
fruits = ["apple", "banana", "orange", "grape"]

print("All fruits:")
for fruit in fruits:
    print(f"  - {fruit}")

print(f"\nTotal fruits: {len(fruits)}")
```

## Dictionary Operations

```python
# Working with dictionaries
student_scores = {
    "Alice": 95,
    "Bob": 87,
    "Charlie": 92,
    "Diana": 88
}

# Calculate average score
total = sum(student_scores.values())
average = total / len(student_scores)

print("Student Scores:")
for name, score in student_scores.items():
    print(f"  {name}: {score}")

print(f"\nClass average: {average:.1f}")
```