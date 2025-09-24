# Algebra Examples

## Quadratic Formula

```python
import math

def quadratic_formula(a, b, c):
    """Solve ax^2 + bx + c = 0"""
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return "No real solutions"
    elif discriminant == 0:
        x = -b / (2*a)
        return f"One solution: x = {x}"
    else:
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        return f"Two solutions: x1 = {x1}, x2 = {x2}"

# Test with x^2 - 5x + 6 = 0
result = quadratic_formula(1, -5, 6)
print(result)
```